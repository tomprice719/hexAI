# predict winning probability for each potential next move

from keras.layers import Conv2D, Input, Add, ZeroPadding2D, Lambda, Multiply, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from keras.activations import relu
import numpy as np
import matplotlib.pyplot as plt
from custom_layers import Gate

depth = 4
breadth = 40

position_tensor = Input(shape=(6, 6, 2))
next_move_tensor = Input(shape=(5, 5, 1))
up_layers = [position_tensor]


def make_up_layer(previous_layer):
    return Conv2D(breadth, 3, padding="same", activation="relu")(previous_layer)


def make_down_layer(up_layer1, up_layer2, down_layer):
    layers = [Conv2D(breadth, 3, padding="same")(up_layer1),
              Gate(-3)(Conv2D(breadth, 3, padding="same")(up_layer2))]
    if down_layer is not None:
        layers.append(Gate(-3)(Conv2D(breadth, 3, padding="same")(down_layer)))
    return Lambda(lambda x: relu(x))(Add()(layers))


for _ in range(depth):
    up_layers.append(make_up_layer(up_layers[-1]))

up_layers.reverse()

down_layer = None

for i, up_layer in enumerate(up_layers[:-1]):
    down_layer = make_down_layer(up_layers[i + 1], up_layer, down_layer)

output_layer = Add()([Conv2D(1, 3, padding="valid")(ZeroPadding2D(((0, 1), (0, 1)))(position_tensor)),
                      Conv2D(1, 3, padding="valid")(ZeroPadding2D(((0, 1), (0, 1)))(down_layer))])

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

validation_size = 10000

model.fit(
    [positions[:-validation_size], moves[:-validation_size]],
    winners[:-validation_size],
    batch_size=32,
    validation_split=0.01,
    epochs=1000,
    shuffle=True
)
