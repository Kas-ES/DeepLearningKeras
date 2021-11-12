import os

from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

physical_devices = tf.config.list_physical_devices("GPU")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# convert from int to float and normalize the pixel range to 0-1 by dividing by 255
x_train = tf.cast(x_train, "float32")
x_test = tf.cast(x_test, "float32")
x_train = x_train / 255.0
x_test = x_test / 255.0


# A MODEL WITH MORE CONV LAYERS AND DENSE LAYERS, AND COPY THAT AND TRY ELU ACTIVATION


def modelMORECONVDENSE():
    inputs = keras.Input(shape=(32, 32, 3))

    #FIRST
    x = layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    #x = layers.Dropout(0.25)(x)

    #SECOND
    x = layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    #x = layers.Dropout(0.25)(x)

    #THIRD
    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    #x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.50)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def model_Regularizer_DropOut():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def Simple_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def resNet50():
    model = keras.Sequential()
    model.add(keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(32,32,3)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.layers[0].trainable=False
    return model


model = resNet50()

print(model.summary())

def dataAugmentConf():
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # brightness_range=(0.2, 0.8)
    )
    return datagen


# Create datagenerator
#datagen = dataAugmentConf()
#datagen.fit(x_train)
#y_train = tf.reshape(y_train, (-1))
#y_train = tf.cast(y_train, tf.float32)
#x_train = tf.cast(x_train, tf.float32)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"],
)
# 0.3e-4
# 3e-4

#history = model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=10, verbose=2,validation_data=(x_test, y_test))
history = model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test, batch_size=64, verbose=2)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
