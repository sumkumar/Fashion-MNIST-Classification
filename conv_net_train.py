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


cnn_model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu',
           input_shape=shape_tuple, kernel_initializer='he_normal'),
    MaxPooling2D(pool_size=2),
    Dropout(0.25),
    Conv2D(64, kernel_size=3, activation='relu'),
    Dropout(0.25),
    Conv2D(128, kernel_size=3, activation='relu'),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])


lr = 0.001
batch_size = 512
epochs = 50
itr = 400
plot_x = np.ndarray((itr), dtype=float)
plot_loss = np.ndarray((itr), dtype=float)
plot_acc = np.ndarray((itr), dtype=float)

for i in range(itr) :
    plot_x[i] = batch_size
    cnn_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )
    cnn_model.fit(
        x_train, y_train, batch_size=batch_size,
        epochs = epochs, verbose=0,
        validation_data=(x_validate, y_validate)
    )
    score = cnn_model.evaluate(x_validate, y_validate, verbose=0)
    plot_acc[i] = score[1]
    plot_loss[i] = score[0]
    #print(lr)
    #print(score[1])
    #print(score[0])
    lr += 0.0001

print(print_list(plot_x))
print(print_list(plot_acc))
print(print_list(plot_loss))