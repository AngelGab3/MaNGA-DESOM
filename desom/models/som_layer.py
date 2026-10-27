"""
SOMLayer: Self-Organizing Map layer for TensorFlow / Keras.
Implements 2D rectangular grid topology, pairwise distance computation,
and Gaussian neighborhood weighting.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputSpec, Layer


def som_loss(weights: tf.Tensor, distances: tf.Tensor) -> tf.Tensor:
    """
    SOM reconstruction loss: weighted sum of prototype distances.
    L_som = mean_i ( sum_j w_ij * ||z_i - m_j||^2 )
    """
    w = tf.cast(weights, tf.float32)
    d = tf.cast(distances, tf.float32)
    return tf.reduce_mean(tf.reduce_sum(w * d, axis=1))


class SOMLayer(Layer):
    """
    Self-Organizing Map layer with 2D rectangular topology.
    Maps latent representation z (batch_size, latent_dim) to pairwise squared
    euclidean distances (batch_size, n_prototypes).
    """

    def __init__(
        self,
        map_size: tuple[int, int] = (15, 15),
        prototypes: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.map_size = tuple(map_size)
        self.n_prototypes = self.map_size[0] * self.map_size[1]
        self.initial_prototypes = prototypes
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.prototypes = self.add_weight(
            shape=(self.n_prototypes, input_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="prototypes",
        )
        if self.initial_prototypes is not None:
            self.set_weights([self.initial_prototypes])
            self.initial_prototypes = None
        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Computes squared euclidean distances:
        d_ij = ||z_i - m_j||^2
        """
        # Shape: (batch_size, 1, latent_dim) - (1, n_prototypes, latent_dim)
        diff = tf.expand_dims(inputs, axis=1) - self.prototypes
        return tf.reduce_sum(tf.square(diff), axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_prototypes)

    def get_config(self):
        config = super().get_config()
        config.update({"map_size": self.map_size})
        return config


# ============================================================================
# Topographic Distance & Neighborhood Utilities
# ============================================================================
MAP_DISTANCE_MODES = ("manhattan", "paper")


def grid_distance_matrix(map_size: tuple[int, int], mode: str = "manhattan") -> np.ndarray:
    """
    Pairwise distance between every pair of SOM nodes on the rectangular grid.
    Returns D with shape (n_prototypes, n_prototypes), D[c, j] = distance(node c, node j).

    mode:
      - "manhattan": exact |row_c - row_j| + |col_c - col_j|.
      - "paper": replicates the original code (Forest et al. / Teimoorinia et al.), where the
        row term is approximated by |c - j| // width. It differs from the exact value for nodes
        that are in adjacent rows but whose flat indices differ by less than `width`.
    """
    if mode not in MAP_DISTANCE_MODES:
        raise ValueError(f"Unknown map distance mode '{mode}'. Choose one of {MAP_DISTANCE_MODES}.")
    h, w = int(map_size[0]), int(map_size[1])
    n = h * w
    c = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    d_col = np.abs(c % w - j % w)
    if mode == "paper":
        d_row = np.abs(c - j) // w
    else:
        d_row = np.abs(c // w - j // w)
    return (d_row + d_col).astype(np.int32)


def compute_map_distances(
    y_pred: np.ndarray, map_size: tuple[int, int], mode: str = "manhattan"
) -> np.ndarray:
    """
    Grid distances between the assigned BMU and all SOM nodes.
    y_pred: array of node indices (N,)
    Returns: distance matrix (N, n_prototypes)
    """
    return grid_distance_matrix(map_size, mode)[np.asarray(y_pred, dtype=int)]


def neighborhood_function(dist: np.ndarray, T: float) -> np.ndarray:
    """
    Gaussian neighborhood function: h_cj(T) = exp(-dist_cj^2 / T^2)
    """
    return np.exp(-(dist**2) / np.maximum(T**2, 1e-8)).astype(np.float32)


def get_temperature(
    iteration: int, som_iterations: int, Tmax: float = 15.0, Tmin: float = 0.5
) -> float:
    """
    Exponential cooling schedule for SOM neighborhood width.
    After `som_iterations` the temperature is kept at Tmin (as in the original code).
    """
    if iteration >= som_iterations:
        return float(Tmin)
    return float(Tmax * (Tmin / Tmax) ** (iteration / max(som_iterations - 1, 1)))
