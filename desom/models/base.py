"""
Shared logic for DESOM-1 and DESOM-2.

A DESOM couples an autoencoder with a SOMLayer attached to the latent space. Everything that does
not depend on the autoencoder architecture (prototype initialisation, encode / decode, BMU search,
neighbourhood weights, weight save / load, optimizer state) lives here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .som_layer import SOMLayer, grid_distance_matrix, som_loss


class BaseDESOM:
    """
    Subclasses must call `_assemble(autoencoder, encoder, decoder, som_name)` from `__init__`
    after setting `map_size`, `gamma`, `lr` and `map_distance`.
    """

    map_size: Tuple[int, int]
    gamma: float
    lr: float
    map_distance: str = "manhattan"

    # ------------------------------------------------------------------ assembly
    def _assemble(self, autoencoder: Model, encoder: Model, decoder: Model, som_name: str) -> None:
        self.n_prototypes = self.map_size[0] * self.map_size[1]
        self.autoencoder, self.encoder, self.decoder = autoencoder, encoder, decoder
        self.som_layer = SOMLayer(map_size=self.map_size, name=som_name)

        som_out = self.som_layer(self.encoder.output)
        # Joint model: input -> [reconstruction, squared distances to prototypes]
        self.model = Model(
            inputs=self.autoencoder.input,
            outputs=[self.autoencoder.output, som_out],
            name=f"DESOM_{som_name}",
        )
        # Light model that only returns distances (no decoder pass); shares all weights.
        self.bmu_model = Model(inputs=self.autoencoder.input, outputs=som_out, name="bmu_model")

        self.set_map_distance(self.map_distance)
        self.compile()

    def set_map_distance(self, mode: str) -> None:
        """Selects how grid distance between SOM nodes is measured ("manhattan" or "paper")."""
        self.map_distance = mode
        self.grid_dist = tf.constant(grid_distance_matrix(self.map_size, mode), dtype=tf.float32)

    def compile(self) -> None:
        """Compiles with MSE reconstruction and gamma-weighted SOM loss."""
        self.model.compile(
            loss=["mse", som_loss],
            loss_weights=[1.0, self.gamma],
            optimizer=Adam(learning_rate=self.lr),
        )

    # ------------------------------------------------------------------ SOM helpers
    def init_som_weights(self, X_sample: np.ndarray, seed: int = 42) -> None:
        """Initializes prototype vectors by randomly sampling encoded training data."""
        rng = np.random.RandomState(seed)
        replace = len(X_sample) < self.n_prototypes
        idx = rng.choice(len(X_sample), size=self.n_prototypes, replace=replace)
        self.som_layer.set_weights([self.encode(X_sample[idx])])

    def encode(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        return self.encoder.predict(x, batch_size=batch_size, verbose=0)

    def decode(self, z: np.ndarray, batch_size: int = 512) -> np.ndarray:
        return self.decoder.predict(z, batch_size=batch_size, verbose=0)

    def distances(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Squared latent distances (N, n_prototypes) to every prototype."""
        return self.bmu_model.predict(x, batch_size=batch_size, verbose=0)

    def predict_bmu(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Returns Best Matching Unit index (0 .. n_prototypes-1) for each input."""
        return np.argmin(self.distances(x, batch_size=batch_size), axis=1)

    @tf.function(reduce_retracing=True)
    def _bmu_weights_graph(self, x: tf.Tensor, T: tf.Tensor):
        d = self.bmu_model(x, training=False)
        bmu = tf.argmin(d, axis=1)
        dist = tf.gather(self.grid_dist, bmu)
        w = tf.exp(-tf.square(dist) / tf.maximum(tf.square(T), 1e-8))
        return d, w

    def bmu_weights(self, x: np.ndarray, T: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes (distances, neighbourhood weights) for a batch inside a compiled graph.
        This replaces the per-iteration `model.predict` of the original code.
        """
        d, w = self._bmu_weights_graph(tf.convert_to_tensor(x, dtype=tf.float32), tf.constant(T, tf.float32))
        return d.numpy(), w.numpy()

    @property
    def prototypes(self) -> np.ndarray:
        return self.som_layer.get_weights()[0]

    # ------------------------------------------------------------------ persistence
    @staticmethod
    def _resolve_weights_path(path: Union[str, Path]) -> Path:
        """Keras 3 requires '*.weights.h5'. A plain '*.h5' name is mapped to it."""
        p = Path(path)
        if str(p).endswith(".h5") and not str(p).endswith(".weights.h5"):
            return p.with_name(p.name[:-3] + ".weights.h5")
        return p

    def save_weights(self, path: Union[str, Path], keep_plain_copy: bool = False) -> None:
        """
        Saves to '<name>.weights.h5'. Set keep_plain_copy to also write '<name>.h5'
        (kept only for tools that look for the legacy filename).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        wp = self._resolve_weights_path(p)
        self.model.save_weights(str(wp))
        if keep_plain_copy and wp != p:
            shutil.copy2(wp, p)

    def load_weights(self, path: Union[str, Path]) -> None:
        """Loads '<name>.weights.h5' if present, otherwise the given path."""
        p = Path(path)
        wp = self._resolve_weights_path(p)
        self.model.load_weights(str(wp if wp.exists() else p))

    # ------------------------------------------------------------------ optimizer state
    def save_optimizer_state(self, path: Union[str, Path]) -> None:
        opt = self.model.optimizer
        variables = getattr(opt, "variables", None)
        if variables is None or not len(variables):
            return
        np.savez(str(path), **{f"v{i}": np.asarray(v.numpy()) for i, v in enumerate(variables)})

    def load_optimizer_state(self, path: Union[str, Path]) -> bool:
        """Restores Adam moments / step count. Returns False if shapes do not match."""
        p = Path(path)
        if not p.exists():
            return False
        opt = self.model.optimizer
        try:
            opt.build(self.model.trainable_variables)
            with np.load(p) as data:
                if len(data.files) != len(opt.variables):
                    return False
                for i, v in enumerate(opt.variables):
                    arr = data[f"v{i}"]
                    if tuple(arr.shape) != tuple(v.shape):
                        return False
                for i, v in enumerate(opt.variables):
                    v.assign(data[f"v{i}"])
            return True
        except Exception:
            return False
