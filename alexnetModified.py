""" 

Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.
Source: tflearn
https://github.com/tflearn/tflearn/edit/master/examples/images/alexnet.py
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.

Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)


# Credit: https://github.com/edward0im/Alexnet-using-keras-2x-Tensorflow/edit/master/alexnet.py
# Using Keras 2.1.2 (180116)
# haven't tested enough. PLEASE use it carefully because I just wrote this code in 5 minutes :-)
"""

# Modified AlexNet from the above reference for TF 2.1 along with adjustments for our 80x60 grayscale image size
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense


def alexnet():
    model = keras.Sequential()
    
    # C1
    model.add(Conv2D(filters = 96, kernel_size = (12, 12), strides=(4, 4), padding='valid', input_shape=(80,60,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    # C2
    model.add(Conv2D(256, (2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3, 2), strides=(2, 2)))

    # C3
    model.add(Conv2D(384, (3, 3), padding='same'))

    # C4
    model.add(Conv2D(384, (3, 3), padding='same'))

    # C5
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    return model
