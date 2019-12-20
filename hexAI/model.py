# predict single winning probability for a board position

from keras.layers import Conv2D, Input, GlobalAveragePooling2D, Dense, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
import itertools
from .config import board_size, main_model_args, main_model_location
from .position_utils import input_names
import numpy as np


class RandomModel():
    def predict(self, model_input):
        shape = set(model_input[k].shape for k in input_names.values())
        assert (len(shape) == 1)
        return np.random.uniform(-1.0, 1.0, shape.pop()[0])


def create_model(depth=5, breadth=20, learning_rate=0.001):
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
    model = create_model(**main_model_args)
    model.load_weights(main_model_location)
    return model
