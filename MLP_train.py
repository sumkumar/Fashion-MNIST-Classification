import numpy as np
import sklearn as sk
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def print_list(arr):
    s = ""
    for i in arr:
        s += str(float(i)) + ", "
    return s


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
shape_tuple = (28, 28, 1)
train_images = train_images.reshape((train_images.shape[0], *shape_tuple))
test_images = test_images.reshape((test_images.shape[0], *shape_tuple))
x_train, x_validate, y_train, y_validate = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=12345,
)


nn_model = Sequential([
    Flatten(),
    Dense(8192, activation='relu'),
    Dropout(0.25),
    Dense(1024, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.35),
    Dense(10, activation='softmax')
])