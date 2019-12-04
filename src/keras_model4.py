# predict single winning probability for a board position

from keras.layers import Conv2D, Input, GlobalAveragePooling2D, Dense, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
import itertools
from utils import input_names


def create_model(depth=5, breadth=40):
    input_tensors = [Input(shape=(6, 6, 2), name=input_names[k]) for k in itertools.product((0, 1), (False, True))]
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

    output_tensor = Add()(out_components)

    model = Model([input_tensors], [output_tensor])

    optimizer = Adam(lr=0.0001)

    model.compile(
        loss=BinaryCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=[BinaryAccuracy(threshold=0.0)]
    )
    return model
