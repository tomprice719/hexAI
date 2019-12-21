"""Functions for creating models to predict whether the current player in a game of hex will win."""

from keras.layers import Conv2D, Input, GlobalAveragePooling2D, Dense, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
import itertools
from .config import board_size, main_model_args, main_model_location
from .model_input import input_names
import numpy as np


class RandomModel:
    """
    A model that selects moves randomly.
    """

    def predict(self, model_input):
        shape = set(model_input[k].shape for k in input_names.values())
        assert (len(shape) == 1)
        return np.random.uniform(-1.0, 1.0, shape.pop()[0])


def create_model(depth=5, breadth=20, learning_rate=0.001):
    """
    A model for predicting whether the current player in a game of hex will win.
    Applies a a sequence of convolutional layers to the input. Each convolutional unit is average-pooled,
    then a dense layer connects these pools to the output.
    The network is provided four representations of the current state of the board,
    by applying 180 degree rotational symmetry and diagonal reflection + player swapping symmetry.
    The second symmetry is not quite a true symmetry since swapping players also changes who the current player is.
    For this reason, the final dense layer has different weights for the player-swapped inputs, so that this difference
    can be taken into account. The convolutional layers use the same weights in all four cases.
    Output of the network should be interpreted as a logit
    representing the probability that the current player will win.
    Functions for constructing input data for these models can be found in the model_input module.

    depth: number of convolutional layers applied to each of the four representations of the board state.
    breadth: number of units in each convolutional layer.
    learning_rate: learning rate for Adam optimizer.
    """
    input_tensors = [Input(shape=(board_size + 1, board_size + 1, 2), name=input_names[k])
                     for k in itertools.product((0, 1), (False, True))]
    out_components = []
    tensors = input_tensors
    pool = GlobalAveragePooling2D()

    for i in range(depth):
        conv_layer = Conv2D(breadth, 3, padding="same", activation="relu")
        tensors = list(map(conv_layer, tensors))
        dense_layer = Dense(1, kernel_initializer="zeros")
        out_components += [dense_layer(pool(t)) for t in tensors[:2]]
        dense_layer = Dense(1, kernel_initializer="zeros")
        out_components += [dense_layer(pool(t)) for t in tensors[2:]]

    output_tensor = Add(name="winners")(out_components)

    model = Model(input_tensors, [output_tensor])

    optimizer = Adam(lr=learning_rate)

    model.compile(
        loss=BinaryCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=[BinaryAccuracy(threshold=0.0)]
    )
    return model


def get_main_model():
    """
    Creates and returns the main model specified in config.py.
    """
    model = create_model(**main_model_args)
    model.load_weights(main_model_location)
    return model
