# predict winning probability for each potential next move

from keras.layers import Conv2D, Input, Add, ZeroPadding2D, Lambda, Multiply, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
import numpy as np
import matplotlib.pyplot as plt

depth = 6
breadth = 80

position_tensor = Input(shape=(6, 6, 2))
next_move_tensor = Input(shape=(5, 5, 1))
up_layers = [position_tensor]


def make_up_layer(previous_layer):
    return Conv2D(breadth, 3, padding="same", activation="relu")(previous_layer)

for _ in range(depth):
    up_layers.append(make_up_layer(up_layers[-1]))

output_layer = Add()([Conv2D(1, 3, padding="valid", kernel_initializer="zeros")(ZeroPadding2D(((0, 1), (0, 1)))(layer))
                      for layer in up_layers])

scalar_output = GlobalMaxPooling2D()(Multiply()([output_layer, next_move_tensor]))

model = Model([position_tensor, next_move_tensor], [scalar_output])

optimizer = Adam(lr=0.001)

model.compile(
    loss=BinaryCrossentropy(from_logits=True),
    optimizer=optimizer,
    metrics=[BinaryAccuracy(threshold=0)]
)

data = np.load("training_data3.npz")
positions = data["positions"]
moves = data["moves"]
winners = data["winners"]

fig = plt.figure(figsize=(8, 8))

model.fit(
    [positions, moves],
    winners,
    batch_size=16,
    validation_split=0.01,
    epochs=1000,
    shuffle=True
)
