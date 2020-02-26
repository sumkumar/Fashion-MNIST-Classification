"""
Code to use the saved models for testing
"""

import numpy as np
import pdb
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras


def test(model, test_images, test_labels):
    """ Training the model using the given dataloader for 1 epoch.

    Input: Model, Dataset, optimizer,
    """

    loss, acc = model.evaluate(test_images, test_labels)
    ypred = model.predict(test_images)
    ypred = np.argmax(ypred, axis=1)

    return loss, test_labels, ypred


if __name__ == "__main__":

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        MLP_model_name = 'models/MLP_model_CPU.h5'
        convnet_model_name = 'models/convnet_model_CPU.h5'
        raise SystemError('GPU device not found')
    else :
        MLP_model_name = 'models/MLP_model_GPU.h5'
        convnet_model_name = 'models/convnet_model_GPU.h5'
    print('Found GPU at: {}'.format(device_name))

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    model_MLP = tf.keras.models.load_model(MLP_model_name)
    model_conv_net = tf.keras.models.load_model(convnet_model_name)

    loss, gt, pred = test(model_MLP, test_images, test_labels)
    with open("multi-layer-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))

    loss, gt, pred = test(model_conv_net, test_images, test_labels)
    with open("convolution-neural-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))
