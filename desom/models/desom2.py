"""
DESOM-2: Deep Embedded Self-Organizing Map for Galaxy Fingerprints.
Maps 2D spectral fingerprints (15x15) onto a 10x10 galaxy diversity SOM grid.
"""

from __future__ import annotations

from typing import Tuple

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Reshape
from tensorflow.keras.models import Model

from .base import BaseDESOM


def build_cnn_2dae(
    input_shape: Tuple[int, int, int] = (15, 15, 1),
    latent_dims: int = 128,
) -> Tuple[Model, Model, Model]:
    """
    Constructs the 2D Convolutional Autoencoder for DESOM-2 from Archinuk et al.
    Returns (autoencoder, encoder, decoder).
    """
    input_layer = Input(shape=input_shape, name="fingerprint_input")
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="enc_conv")(input_layer)

    shape_before_flatten = K.int_shape(x)[1:]
    num_neurons = int(shape_before_flatten[0] * shape_before_flatten[1] * shape_before_flatten[2])

    x = Flatten()(x)
    encoded = Dense(latent_dims, name="z_fingerprint")(x)

    encoder = Model(input_layer, encoded, name="encoder_2d")

    # Decoder
    encoded_input = Input(shape=(latent_dims,), name="encoded_fp_input")
    x = Dense(num_neurons, activation="linear", name="dec_dense")(encoded_input)
    x = Reshape(shape_before_flatten, name="dec_reshape")(x)
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="dec_conv1")(x)
    decoded = Conv2D(1, (1, 1), activation="linear", padding="same", name="reconstruction_2d")(x)

    decoder = Model(encoded_input, decoded, name="decoder_2d")

    full_decoded = decoder(encoder(input_layer))
    autoencoder = Model(input_layer, full_decoded, name="autoencoder_2d")

    return autoencoder, encoder, decoder


class DESOM2(BaseDESOM):
    """
    Complete DESOM-2 Model: 2D Autoencoder + 10x10 SOMLayer.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (15, 15, 1),
        map_size: Tuple[int, int] = (10, 10),
        latent_dims: int = 128,
        gamma: float = 8e-6,
        lr: float = 1e-4,
        map_distance: str = "manhattan",
    ):
        self.input_shape = tuple(input_shape)
        self.map_size = tuple(map_size)
        self.latent_dims = latent_dims
        self.gamma = gamma
        self.lr = lr
        self.map_distance = map_distance

        autoencoder, encoder, decoder = build_cnn_2dae(
            input_shape=self.input_shape, latent_dims=self.latent_dims
        )
        self._assemble(autoencoder, encoder, decoder, som_name="SOM_2D")
