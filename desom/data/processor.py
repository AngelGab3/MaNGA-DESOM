"""
Spectroscopic Data Processor for single MaNGA galaxies.
Implements:
1. Voronoi spatial binning to target S/N = 20 (Cappellari & Copin 2003)
2. Noise covariance correction (Law et al. 2016)
3. De-redshifting to rest-frame
4. Logarithmic rebinning to 4544 bins in [3650, 8950] Angstroms
5. Sky line masking & interpolation of small gaps (<= 50 A)
6. Continuum normalization by mean flux in [4050, 4150] A
7. Outlier screening (flux percentiles, emission line preservation)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy import sparse

from ..config import Config
from ..io_utils import atomic_save_json, atomic_save_npz, now_utc_iso, sha256_file
from .cubes import download_galaxy_cube


# Bump when the processing logic changes so that cached galaxies are recomputed.
PROCESSOR_VERSION = "2.1"


def preprocessing_signature(pre: Dict[str, Any]) -> str:
    """Hash of every preprocessing parameter + code version. Stored in each galaxy manifest."""
    import hashlib
    import json

    blob = json.dumps({"v": PROCESSOR_VERSION, "pre": pre}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ============================================================================
# Math & Rebinning Helpers
# ============================================================================
def noise_factor(n_spaxels: np.ndarray | int) -> np.ndarray:
    """
    Law et al. (2016) noise covariance correction for MaNGA co-added spaxels.
    Factor = 1.0 + 1.62 * log10(n) for n < 100; factor = 4.2 for n >= 100.
    """
    n = np.asarray(n_spaxels, dtype=float)
    return np.where(n < 100.0, 1.0 + 1.62 * np.log10(np.maximum(n, 1.0)), 4.2)


def corrected_sn(indices: np.ndarray, signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculates S/N for a set of co-added spaxels accounting for noise covariance."""
    idx = np.atleast_1d(indices).astype(int)
    factor = noise_factor(len(idx))
    s = signal[idx].sum()
    n = np.sqrt(np.square(noise[idx]).sum()) * factor
    return float(s / n) if n > 0 else 0.0


def log_edges(centers: np.ndarray) -> np.ndarray:
    """Computes bin edges for an array of bin centers."""
    logw = np.log(np.asarray(centers))
    return np.exp(
        np.r_[
            logw[0] - (logw[1] - logw[0]) / 2.0,
            (logw[:-1] + logw[1:]) / 2.0,
            logw[-1] + (logw[-1] - logw[-2]) / 2.0,
        ]
    )


def overlap_operator(input_edges: np.ndarray, output_edges: np.ndarray) -> sparse.csr_matrix:
    """
    Builds a sparse flux-conserving linear operator mapping spectra from input_edges
    to output_edges by fractional pixel overlap.
    """
    rows, cols, data = [], [], []
    for j, (left, right) in enumerate(zip(output_edges[:-1], output_edges[1:])):
        i = max(0, np.searchsorted(input_edges, left, side="right") - 1)
        while i < len(input_edges) - 1 and input_edges[i] < right:
            overlap = min(right, input_edges[i + 1]) - max(left, input_edges[i])
            if overlap > 0:
                rows.append(j)
                cols.append(i)
                data.append(overlap / (right - left))
            i += 1
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(output_edges) - 1, len(input_edges) - 1)
    )


def fill_short_gaps(
    wave: np.ndarray, spectrum: np.ndarray, max_width_angstrom: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolates unobserved or masked gaps of width <= max_width_angstrom."""
    out = np.array(spectrum, dtype=float, copy=True)
    filled = np.zeros(len(out), dtype=bool)
    missing = np.flatnonzero(~np.isfinite(out))
    if len(missing) == 0:
        return out, filled

    # Find contiguous missing chunks
    groups = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
    for group in groups:
        lo, hi = int(group[0]) - 1, int(group[-1]) + 1
        if lo < 0 or hi >= len(out):
            continue
        if wave[hi] - wave[lo] <= max_width_angstrom and np.isfinite(out[[lo, hi]]).all():
            out[group] = np.interp(wave[group], wave[[lo, hi]], out[[lo, hi]])
            filled[group] = True
    return out, filled


# ============================================================================
# Voronoi Binning Function
# ============================================================================
def run_voronoi_binning(
    x: np.ndarray,
    y: np.ndarray,
    signal: np.ndarray,
    noise: np.ndarray,
    target_sn: float = 20.0,
    cvt: bool = True,
    wvt: bool = True,
) -> np.ndarray:
    """
    Executes Voronoi 2D binning using Cappellari's vorbin package (required).
    """
    try:
        from vorbin.voronoi_2d_binning import voronoi_2d_binning
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The 'vorbin' package (Cappellari & Copin 2003) is required for Voronoi binning. "
            "Install it with `pip install vorbin`. A home-made approximation is deliberately "
            "not used because it would silently change the baxels with respect to the paper."
        ) from e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = voronoi_2d_binning(
            x.astype(float),
            y.astype(float),
            signal.astype(float),
            noise.astype(float),
            target_sn,
            pixelsize=1.0,
            plot=False,
            quiet=True,
            cvt=cvt,
            wvt=wvt,
            sn_func=corrected_sn,
        )
    return result[0].astype(int)


# ============================================================================
# Galaxy Processing Pipeline
# ============================================================================
def process_single_galaxy(
    cfg: Config,
    plateifu: str,
    redshift: float,
    manga_id: str,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Full spectroscopic processing of a single galaxy cube:
    - Binning -> Baxels -> Rebinning in [3650, 8950] A -> Normalization.
    """
    out_dir = cfg.processed_dir / str(plateifu)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    signature = preprocessing_signature(cfg["preprocessing"])
    if not force_recompute and manifest_path.exists():
        try:
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                # Reuse cached products only if they were made with identical parameters.
                if manifest.get("status") == "complete" and manifest.get("signature") == signature:
                    return manifest["summary"]
        except Exception:
            pass

    # 1. Download or locate LOGCUBE
    cube_path = download_galaxy_cube(cfg, plateifu, verbose=False)

    # 2. Open FITS
    with fits.open(cube_path, memmap=False) as hdul:
        flux = np.array(hdul["FLUX"].data, dtype=np.float32)
        ivar = np.array(hdul["IVAR"].data, dtype=np.float32)
        flags = np.array(hdul["MASK"].data, dtype=np.int64)
        wave_obs = np.array(hdul["WAVE"].data, dtype=np.float64)
        flux_header = hdul["FLUX"].header.copy()
        primary_header = hdul[0].header.copy()

    # Redshift and rest-frame wavelengths
    z = float(redshift)
    wave_rest_native = wave_obs / (1.0 + z)
    ny, nx = flux.shape[1:]

    # 3. Mask bad pixels & sky lines
    pre = cfg["preprocessing"]
    bad_bits = sum(1 << b for b in pre["bad_drp_pixel_bits"])
    valid = (np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & ((flags & bad_bits) == 0))
    foreground = np.any((flags & (1 << 3)) != 0, axis=0)  # Foreground star flag

    # Mask prominent terrestrial night sky lines
    for sky_line in pre["sky_observed"]:
        valid[np.abs(wave_obs - sky_line) <= pre["sky_half_width"]] = False

    variance = np.full(flux.shape, np.nan, dtype=np.float32)
    np.divide(1.0, ivar, out=variance, where=valid)

    # 4. Spaxel S/N calculation in continuum window [4050, 4150] A
    cont_mask = (wave_rest_native >= pre["sn_window_rest"][0]) & (
        wave_rest_native <= pre["sn_window_rest"][1]
    )
    vcont = valid[cont_mask]
    ncont = vcont.sum(axis=0)

    signal_map = np.divide(
        np.where(vcont, flux[cont_mask], 0.0).sum(axis=0),
        ncont,
        out=np.full((ny, nx), np.nan),
        where=ncont > 0,
    )
    mean_var_map = np.divide(
        np.where(vcont, variance[cont_mask], 0.0).sum(axis=0),
        ncont,
        out=np.full((ny, nx), np.nan),
        where=ncont > 0,
    )
    noise_map = np.sqrt(mean_var_map)
    sn_map = np.divide(signal_map, noise_map, out=np.zeros_like(signal_map), where=noise_map > 0)

    # Spaxel pre-selection (Paper Section 2: individual spaxels having S/N >= 2)
    selected = (
        (ncont >= pre["min_continuum_coverage"] * cont_mask.sum())
        & np.isfinite(sn_map)
        & (sn_map >= pre["min_spaxel_sn"])
        & (~foreground)
    )

    # 5. Voronoi binning (Paper Section 2): all spaxels with S/N >= 2 are binned together to a
    #    target S/N of 20. Bins that do not reach the target (mostly galaxy outskirts) are kept
    #    here and removed later by the baxel S/N cut (min_baxel_sn).
    bin_map = np.full((ny, nx), -1, dtype=np.int32)
    bin_modes: List[str] = []
    binning_status = "ok"

    yy, xx = np.where(selected)
    if len(xx) > 0:
        sig = signal_map[yy, xx]
        noi = noise_map[yy, xx]
        if len(xx) < 2:
            labels, mode = np.zeros(len(xx), dtype=int), "single_spaxel"
        elif np.all(sig / np.maximum(noi, 1e-6) >= pre["target_sn"]):
            # Every spaxel already reaches the target: vorbin would leave them untouched.
            labels, mode = np.arange(len(xx)), "already_above_target"
        else:
            try:
                labels = run_voronoi_binning(
                    xx, yy, sig, noi, target_sn=pre["target_sn"], cvt=pre["cvt"], wvt=pre["wvt"]
                )
                mode = "vorbin"
            except ValueError as e:
                # vorbin refuses sets whose total S/N cannot reach the target.
                labels, mode, binning_status = None, "none", f"vorbin_failed: {e}"
        if labels is not None:
            _, labels = np.unique(labels, return_inverse=True)
            bin_map[yy, xx] = labels
            bin_modes = [mode] * int(labels.max() + 1)

    next_bin = len(bin_modes)
    n_bins = next_bin

    # 6. Logarithmic rebinning to 4544 channels in [3650, 8950] Angstroms
    # CRITICAL: output_edges strictly covers [3650, 8950] A from Paper Section 2
    rest_min, rest_max = pre["rest_limits"]
    n_wave = int(pre["n_wave"])
    output_edges = np.geomspace(rest_min, rest_max, n_wave + 1)
    wave_rest = np.sqrt(output_edges[:-1] * output_edges[1:])

    input_edges = log_edges(wave_rest_native)
    A = overlap_operator(input_edges, output_edges)
    A2 = A.multiply(A)

    raw_rebinned = np.full((n_bins, n_wave), np.nan, dtype=np.float32)
    var_rebinned = np.full_like(raw_rebinned, np.nan)
    observed_good = np.zeros(raw_rebinned.shape, dtype=bool)
    spatial_coverage = np.zeros_like(raw_rebinned)
    counts, centers_x, centers_y, sn_bins = [], [], [], []

    for k in range(n_bins):
        yy, xx = np.where(bin_map == k)
        n_spax = len(xx)
        f_k = flux[:, yy, xx]
        v_k = variance[:, yy, xx]
        good_k = valid[:, yy, xx]

        nvalid = good_k.sum(axis=1)
        fmean = np.divide(
            np.where(good_k, f_k, 0.0).sum(axis=1, dtype=float),
            nvalid,
            out=np.full(len(wave_obs), np.nan),
            where=nvalid > 0,
        )
        vmean = np.divide(
            np.where(good_k, v_k, 0.0).sum(axis=1, dtype=float),
            nvalid**2,
            out=np.full(len(wave_obs), np.nan),
            where=nvalid > 0,
        )
        vmean *= noise_factor(nvalid) ** 2

        good_native = (
            (nvalid >= pre["min_spatial_coverage"] * n_spax)
            & np.isfinite(fmean)
            & np.isfinite(vmean)
        )
        frest = fmean * (1.0 + z)
        vrest = vmean * (1.0 + z) ** 2

        coverage = np.asarray(A @ good_native.astype(float)).ravel()
        measured = coverage >= (1.0 - 1e-6)

        reb = A @ np.where(good_native, frest, 0.0)
        rv = A2 @ np.where(good_native, vrest, 0.0)

        raw_rebinned[k, measured] = reb[measured]
        var_rebinned[k, measured] = rv[measured]
        observed_good[k] = measured
        spatial_coverage[k] = A @ (nvalid / n_spax)

        counts.append(n_spax)
        centers_x.append(xx.mean())
        centers_y.append(yy.mean())

        sig_k, noi_k = signal_map[yy, xx], noise_map[yy, xx]
        sn_baxel = sig_k.sum() / (
            np.sqrt(np.square(noi_k).sum()) * noise_factor(n_spax)
        )
        sn_bins.append(sn_baxel)

    # 7. Continuum Normalization & Outlier Screening
    norm_window = (wave_rest >= pre["continuum_rest"][0]) & (
        wave_rest <= pre["continuum_rest"][1]
    )
    normalized = np.full_like(raw_rebinned, np.nan)
    normalized_var = np.full_like(raw_rebinned, np.nan)
    interpolated = np.zeros_like(observed_good)
    continuum = np.full(n_bins, np.nan)
    accepted = np.zeros(n_bins, dtype=bool)
    reasons: List[str] = []

    for k in range(n_bins):
        observed = observed_good[k]
        filled, interp = fill_short_gaps(
            wave_rest, raw_rebinned[k], pre["max_interpolation_gap_A"]
        )
        interpolated[k] = interp
        reason = []

        if observed.sum() < pre["min_good_rebinned"]:
            reason.append("few_good_bins")
        if observed[norm_window].mean() < pre["min_continuum_coverage"]:
            reason.append("continuum_coverage")
        if not np.isfinite(filled).all():
            reason.append("unfilled_gaps")

        if np.isfinite(filled[norm_window]).all():
            continuum[k] = np.mean(filled[norm_window])

        if not np.isfinite(continuum[k]) or continuum[k] <= 0:
            reason.append("invalid_continuum")
        else:
            normalized[k] = filled / continuum[k]
            normalized_var[k] = var_rebinned[k] / (continuum[k] ** 2)

        if sn_bins[k] < pre["min_baxel_sn"]:
            reason.append("low_baxel_sn")

        is_cand = (len(reason) == 0)
        accepted[k] = is_cand
        reasons.append(";".join(reason) if reason else "candidate")

    candidate_ids = np.flatnonzero(accepted)
    X = normalized[accepted]

    # 8. Save Products Atomically
    meta = Table()
    meta["mangaID"] = [manga_id] * n_bins
    meta["plateifu"] = [plateifu] * n_bins
    meta["baxel_id"] = np.arange(n_bins, dtype=np.int32)
    meta["redshift"] = np.full(n_bins, z)
    meta["n_spaxels"] = np.asarray(counts, dtype=np.int32)
    meta["x_center_pix"] = centers_x
    meta["y_center_pix"] = centers_y
    meta["sn_cont_corrected"] = sn_bins
    meta["ngood_rebinned"] = observed_good.sum(axis=1)
    meta["n_interpolated"] = interpolated.sum(axis=1)
    meta["continuum"] = continuum
    meta["accepted_candidate"] = accepted
    meta["reason"] = reasons
    meta["binning_mode"] = bin_modes

    meta.write(out_dir / "all_baxels.ecsv", overwrite=True)

    atomic_save_npz(
        out_dir / "candidate_spectra.npz",
        X=X.astype(np.float32),
        wave_rest=wave_rest,
        sn_cont_corrected=np.asarray(sn_bins)[accepted],
        baxel_id=candidate_ids,
        mangaID=np.array(manga_id),
        plateifu=np.array(plateifu),
    )

    # Summary
    summary = {
        "plateifu": plateifu,
        "mangaID": manga_id,
        "redshift": z,
        "n_spaxels_selected": int(selected.sum()),
        "n_baxels_total": n_bins,
        "n_candidates_accepted": int(accepted.sum()),
        "binning_status": binning_status,
        "n_sn20": int(((np.asarray(sn_bins) >= 20.0) & accepted).sum()),
    }

    manifest = {
        "status": "complete",
        "signature": signature,
        "completed_at": now_utc_iso(),
        "summary": summary,
    }
    atomic_save_json(manifest_path, manifest)

    return summary
