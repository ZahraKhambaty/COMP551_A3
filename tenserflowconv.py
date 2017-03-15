# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import h5py
import numpy

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data
from tflearn.layers.core import dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle

X = numpy.load('/Users/zahrakhambaty/Downloads/tinyX.npy')# this should have shape (26344, 3, 64, 64)
Y= numpy.load('/Users/zahrakhambaty/Downloads/tinyY.npy') 
X_test= numpy.load('/Users/zahrakhambaty/Downloads/tinyX_test.npy') # (6600, 3, 64, 64)



#shuffle data
X, Y = shuffle(X, Y)

# Real-time data preprocessing and normalizing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()


# one encoded output
# transforming to a binary matrix also binary width is 10 since they are 10 classes
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)



# Creating extra synthetic data by flipping, rotating and blurring the images or Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(max_angle=3.)

#defining the network architecture:

#input is a 32 by 32 image with 3 color channels RGB
# Convolutional network building
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
#Step1:convolution
network = conv_2d(network, 32, 3, activation='relu')
#Step2:max pooling
network = max_pool_2d(network, 2)
#Step3: Convolution
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
#Step6: FULLY connected 512 node neural network
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, run_id='cifar10_cnn')
model.save("bird-classifier.tfl")
print("Network trained and saved")



