# @author Pauline Houlgatte
# based on https://www.tensorflow.org/tutorials/layers

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# cnn
#     convolutional layer
#     pooling layer
#     convolutional layer
#     pooling layer
#     dense layer
#     dense layer (output)

# Convolutional layer
#     extracting 5x5 pixels subregions
#     use ReLU activation function
#     method: conv2d()
#     doc: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
# Pooling layer
#     stride : 2
#     use max pooling
#     method: max_pooling2d()
#     doc: https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling2d
# Dense layer
#     use dropout regularization just for the firt dense layer
#     1 024 neurons / 10 for the output layer (one for each target class - 0 to 9)
#     method: dense()
#     doc: https://www.tensorflow.org/api_docs/python/tf/layers/dense

def convolutionalLayer(input,filter,kernel,activation,padding="same"):
  return tf.layers.conv2d(inputs=input,filters=filter,kernel_size=kernel,
    padding=padding,activation=activation)

def poolingLayer():
  return 

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__":
  tf.app.run() # run the tensorflow application
