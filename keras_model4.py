# predict single winning probability for a board position

from keras.layers import Conv2D, Input, GlobalAveragePooling2D, Dense, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
import numpy as np

depth = 5
breadth = 40

input_tensor = Input(shape=(6, 6, 2))
out_components = []
current_layer = input_tensor

for i in range(depth):
    current_layer = Conv2D(breadth, 3, padding="same", activation="relu")(current_layer)
    out_components.append(Dense(1, kernel_initializer="zeros")
                          (GlobalAveragePooling2D()
                           (current_layer)))

output_tensor = Add()(out_components)

model = Model([input_tensor], [output_tensor])

optimizer = Adam(lr=0.001)

model.compile(
    loss=BinaryCrossentropy(from_logits=True),
    optimizer=optimizer,
    metrics=[BinaryAccuracy(threshold=0)]
)

data = np.load("training_data4.npz")
positions = data["positions"]
winners = data["winners"]

print(winners[:10])

validation_size = 10000

model.fit(
    positions[:-validation_size],
    winners[:-validation_size],
    batch_size=32,
    validation_data=(positions[-validation_size:], winners[-validation_size:]),
    epochs=5,
    shuffle=True
)

predictions = model.predict(positions)
print(predictions[:10])
print(winners[:10])
