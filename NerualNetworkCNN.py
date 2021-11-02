import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train = x_train.reshape(-1, 32*32*3).astype("float32") / 255.0
x_test = x_test.reshape(-1, 32*32*3).astype("float32") / 255.0

