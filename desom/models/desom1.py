"""
DESOM-1: Deep Embedded Self-Organizing Map for 1D Spectra.
Combines 1D Convolutional Autoencoder with a 2D SOMLayer.
"""

from __future__ import annotations

from typing import Tuple

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Flatten,
    Input,
    LeakyReLU,
    MaxPooling1D,
    Reshape,
    UpSampling1D,
)
from tensorflow.keras.models import Model

from .base import BaseDESOM


def build_cnn_1dae(
    input_dims: int = 4544,
    latent_dims: int = 256,
    alpha: float = 0.1,
) -> Tuple[Model, Model, Model]:
    """
    Constructs the 1D Convolutional Autoencoder from Archinuk et al.
    Returns (autoencoder, encoder, decoder).
    """
    # 1. Encoder
    input_layer = Input(shape=(input_dims,), name="input")
    x = Reshape((input_dims, 1))(input_layer)

    # 4 Downsampling blocks (4544 / 16 = 284)
    for _ in range(4):
        x = Conv1D(64, 6, strides=1, padding="same")(x)
        x = LeakyReLU(negative_slope=alpha)(x)
        x = MaxPooling1D(pool_size=2)(x)

    shape_before_flatten = K.int_shape(x)[1:]
    num_neurons = int(shape_before_flatten[0] * shape_before_flatten[1])

    x = Flatten()(x)
    x = Dense(latent_dims)(x)
    encoded = LeakyReLU(negative_slope=alpha, name="z")(x)

    encoder = Model(input_layer, encoded, name="encoder")

    # 2. Decoder
    encoded_input = Input(shape=(latent_dims,), name="encoded_input")
    x = Dense(num_neurons, name="decoder_dense")(encoded_input)
    x = LeakyReLU(negative_slope=alpha)(x)
    x = Reshape(shape_before_flatten)(x)

    # 4 Upsampling blocks
    for i in range(4):
        x = UpSampling1D(2)(x)
        x = Conv1D(64, 6, padding="same")(x)
        x = LeakyReLU(negative_slope=alpha)(x)

    x = Conv1D(1, 1, activation="linear", padding="same")(x)
    decoded = Flatten(name="reconstruction")(x)

    decoder = Model(encoded_input, decoded, name="decoder")

    # 3. Full Autoencoder
    full_decoded = decoder(encoder(input_layer))
    autoencoder = Model(input_layer, full_decoded, name="autoencoder_1d")

    return autoencoder, encoder, decoder


class DESOM1(BaseDESOM):
    """
    Complete DESOM-1 Model: 1D Autoencoder + 2D SOMLayer (15x15 by default).
    """

    def __init__(
        self,
        input_dims: int = 4544,
        map_size: Tuple[int, int] = (15, 15),
        latent_dims: int = 256,
        gamma: float = 5e-4,
        lr: float = 5e-5,
        map_distance: str = "manhattan",
    ):
        self.input_dims = input_dims
        self.map_size = tuple(map_size)
        self.latent_dims = latent_dims
        self.gamma = gamma
        self.lr = lr
        self.map_distance = map_distance

        autoencoder, encoder, decoder = build_cnn_1dae(
            input_dims=self.input_dims, latent_dims=self.latent_dims
        )
        self._assemble(autoencoder, encoder, decoder, som_name="SOM")
