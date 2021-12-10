import os
from random import randrange

import numpy as np
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras
from keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices("GPU")

#Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Hot-encode lables
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#Reshape, normalize data and cast as float32
x_train = x_train.reshape(50000, 32, 32, 3).astype("float32") / 255
x_test = x_test.reshape(10000, 32, 32 ,3).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

#Take 10% drom trianing data to validaiton data
x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]

#Take the mean with the std
mean = np.mean(x_train)
std = np.std(x_train)
x_test = (x_test - mean) / std
x_train = (x_train - mean) / std

#Model modifications.
input_image_shape = 32, 32, 3
inputs = keras.Input(shape=input_image_shape)
filters = 32
hidden_layer_nodes = 64
classes = 10

#<--MODELS###
def model6blockVGGstyle():
    # FIRST
    x = tensorflow.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = tensorflow.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)

    # SECOND
    x = tensorflow.keras.layers.Conv2D(filters * 2, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.Conv2D(filters * 2, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)

    # THIRD
    x = tensorflow.keras.layers.Conv2D(filters * 4, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.Conv2D(filters * 4, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.Conv2D(filters * 16, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)

    # FOURTH
    x = tensorflow.keras.layers.Conv2D(filters * 8, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.Conv2D(filters * 8, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.Conv2D(filters * 16, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)

    # FIFTH
    x = tensorflow.keras.layers.Conv2D(filters * 16, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.Conv2D(filters * 16, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.Conv2D(filters * 16, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)

    x = tensorflow.keras.layers.Flatten()(x)
    #x = layers.Dense(hidden_layer_nodes * 256, activation="relu")(x)
    x = tensorflow.keras.layers.Dense(hidden_layer_nodes * 16, activation="relu")(x)
    x = tensorflow.keras.layers.Dropout(0.5)(x)
    x = tensorflow.keras.layers.Dense(hidden_layer_nodes, activation="relu")(x)
    x = tensorflow.keras.layers.Dropout(0.5)(x)
    outputs = tensorflow.keras.layers.Dense(classes)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def model3blockVGGstyle():
    # FIRST
    x = tensorflow.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = tensorflow.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)

    # SECOND
    x = tensorflow.keras.layers.Conv2D(filters * 2, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.Conv2D(filters * 2, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)

    # THIRD
    x = tensorflow.keras.layers.Conv2D(filters * 4, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.Conv2D(filters * 4, 3, padding="same", activation="relu")(x)
    #x = layers.Conv2D(filters * 4, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)

    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(hidden_layer_nodes * 16, activation="relu")(x)
    x = tensorflow.keras.layers.Dropout(0.5)(x)
    x = tensorflow.keras.layers.Dense(hidden_layer_nodes, activation="relu")(x)
    x = tensorflow.keras.layers.Dropout(0.5)(x)
    outputs = tensorflow.keras.layers.Dense(classes)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def model_Regularizer_DropOut():
    x = tensorflow.keras.layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(inputs)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)

    x = tensorflow.keras.layers.Conv2D(filters * 2, 3, padding="same", activation="relu", kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)

    x = tensorflow.keras.layers.MaxPooling2D()(x)
    x = tensorflow.keras.layers.Conv2D(filters * 4, 3, padding="same", activation="relu", kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)

    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(hidden_layer_nodes, activation="relu", kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
    x = tensorflow.keras.layers.Dropout(0.5)(x)
    outputs = tensorflow.keras.layers.Dense(classes)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def Simple_model():
    x = tensorflow.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)
    x = tensorflow.keras.layers.Conv2D(filters * 2, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)
    x = tensorflow.keras.layers.Conv2D(filters * 4, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(hidden_layer_nodes, activation="relu")(x)
    outputs = tensorflow.keras.layers.Dense(classes)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def pretrainedPureVGG16():
    base_model = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    print(base_model.summary())
    x = base_model(inputs, training=True)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tensorflow.keras.layers.Flatten()(x)

    x = tensorflow.keras.layers.Dense(hidden_layer_nodes * 256, activation="relu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    x = tensorflow.keras.layers.Dense(hidden_layer_nodes * 16, activation="relu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = keras.layers.Dense(classes)(x)
    model = keras.Model(inputs, outputs)
    return model


def pretrainedVGG16toplayers():
    base_model = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3))

    model12 = Model(inputs=base_model.input, outputs=base_model.layers[-9].output)
    print(model12.summary())
    x = model12(inputs, training=True)

    #x = tf.keras.layers.Dropout(0.3)(x)

    x = tensorflow.keras.layers.Conv2D(filters * 8, 3, padding="same", activation="elu")(x)
    x = tensorflow.keras.layers.Conv2D(filters * 8, 3, padding="same", activation="elu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)
    #x = tf.keras.layers.Dropout(0.4)(x)

    x = tensorflow.keras.layers.Conv2D(filters * 16, 3, padding="same", activation="elu")(x)
    x = tensorflow.keras.layers.Conv2D(filters * 16, 3, padding="same", activation="elu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.MaxPooling2D()(x)
    #x = tf.keras.layers.Dropout(0.5)(x)

    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(hidden_layer_nodes * 256, activation="elu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.9)(x)
    x = tensorflow.keras.layers.Dense(hidden_layer_nodes, activation="elu")(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = keras.layers.Dense(classes)(x)
    model = keras.Model(inputs, outputs)
    return model


def pretrainedResnet50():
    base_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    print(base_model.summary())
    x = base_model(inputs, training=True)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(classes)(x)
    model = keras.Model(inputs, outputs)
    return model
###MODELS###--->


# Create datagenerator
def dataAugmentConf():
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.8,
        horizontal_flip=True,
        validation_split=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2
    )
    return datagen

datagen = dataAugmentConf()
#fit for training data
#datagen.fit(x_train)

#Create model
model = pretrainedVGG16toplayers()
print(model.summary())

#Compile loss, optimizer and metric
model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1),
    metrics=["categorical_accuracy"],
)

#Train the model
#history = model.fit(datagen.flow(x_train, y_train, batch_size=64, subset='training'), validation_data=datagen.flow(x_train, y_train, batch_size=8, subset='validation'), epochs=100, verbose=2)
#history = model.fit(datagen.flow(x_train, y_train, batch_size=64), validation_data=(x_val, y_val), epochs=100, verbose=2)
history = model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2, validation_data=(x_val, y_val))

#Evaluate on test
model.evaluate(x_test, y_test, batch_size=64, verbose=2)

# list all data in history
print(history.history.keys())
# summarize history for accuracy

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
