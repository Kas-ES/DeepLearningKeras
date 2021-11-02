import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
print(y_train.shape)

#Flatten the data, using reshape where -1 holds the value of the dimension (in this case 50000)
#and next is flattening the dimmension of 32 32 and 3. Normalize the data instead of 0 to 255,
#making them zero and one for faster training since  Pixel values range from 0 to 255
x_train = x_train.reshape(-1, 32*32*3).astype("float32") / 255.0
x_test = x_test.reshape(-1, 32*32*3).astype("float32") / 255.0

print(x_train.shape)
print(x_test.shape)

# Sequential API (Very convenient, not very flexible)
model = keras.Sequential()
model.add(keras.Input(shape=(32*32*3)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
#Outputlayer
model.add(layers.Dense(10, activation='softmax'))


#Functional API (A bit more flexible, able to handle multiple inputs and outputs)
inputs = keras.Input(shape=(32*32*3))
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(256, activation='relu') (x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

#Print model layers overview
print(model.summary())

#Network configurations
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

#Training of the network
model.fit(x_train,y_train, batch_size=52, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=52, verbose=2)