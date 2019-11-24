# predict moves that will immediately win

from keras.layers import Conv2D, Input, Add, ZeroPadding2D, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.activations import relu
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from custom_layers import Gate


def metric1(y_true, y_pred):
    return 1.0 - K.sum(y_true * K.sigmoid(y_pred)) / K.sum(y_true)


def metric2(y_true, y_pred):
    return K.sum((1 - y_true) * K.sigmoid(y_pred)) / K.sum(1 - y_true)


depth = 4
breadth = 40

input_tensor = Input(shape=(6, 6, 2))
up_layers = [input_tensor]


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

output_layer = Add()([Conv2D(1, 3, padding="valid")(ZeroPadding2D(((0, 1), (0, 1)))(input_tensor)),
                      Conv2D(1, 3, padding="valid")(ZeroPadding2D(((0, 1), (0, 1)))(down_layer))])

model = Model([input_tensor], [output_layer])

optimizer = Adam(lr=0.001)

model.compile(
    loss=BinaryCrossentropy(from_logits=True),
    optimizer=optimizer,
    metrics=[metric1, metric2]
)

data = np.load("training_data2.npz")
positions = data["positions"]
winning_moves = data["winning_moves"]

fig = plt.figure(figsize=(8, 8))

validation_size = 10000

model.fit(
    positions[:-validation_size],
    winning_moves[:-validation_size],
    batch_size=32,
    validation_data=(positions[-validation_size:], winning_moves[-validation_size:]),
    epochs=1000,
    shuffle=True
)

predictions = model.predict(positions[:10])

for i in range(5):
    fig.add_subplot(4, 5, i + 1)
    plt.imshow(positions[i, :, :, 0])
for i in range(5):
    fig.add_subplot(4, 5, i + 6)
    plt.imshow(positions[i, :, :, 1])
for i in range(5):
    fig.add_subplot(4, 5, i + 11)
    plt.imshow(winning_moves[i, :, :, 0])
for i in range(5):
    fig.add_subplot(4, 5, i + 16)
    plt.imshow(predictions[i, :, :, 0])

plt.show()
