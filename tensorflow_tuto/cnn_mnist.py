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
#     doc: https://www.tensorflow.org/api_docs/python/tf/layers/conv2d
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

def convolutionalLayer(input,filter,size,activation,padding="same"):
  """
  allow to create a convolutional layer
  @input tensor input
  @filter the number of filters in the convolution (int)
  @size size of the filter (int - when the width and the height have the same value -
   or list)
  @activation the activation function - None for linear activation
  @padding "same" (the output should have the same width and height as input tensor)
   or "valid" - default value : "same"
  """
  return tf.layers.conv2d(inputs=input,filters=filter,kernel_size=size,
    padding=padding,activation=activation)

def poolingLayer(input,size,stride,padding="valid"):
  """
  allow to create a pooling layer
  @input the output of the previous layer
  @size size of the pooling filter (int - same dimensions - or list)
  @stride stride of the pooling (int or list)
  @padding "same" or "valid" - default value : "valid"
  """
  return tf.layers.max_pooling2d(inputs=input,pool_size=size,
    strides=stride,padding=padding)

def denseLayer(input,unit,activation):
  """
  allow to create a dense layer
  @input tensor input
  @unit dimension of the output space (int or long)
  @activation the activation function - None for linear activation
  """
  return tf.layers.dense(inputs=input,units=unit,activation=activation)

def dropout(input,rate,training):
  """
  apply dropout to the input
  doc: https://www.tensorflow.org/api_docs/python/tf/layers/dropout
  @input tensor input
  @rate the dropout rate (0 - 1)
  @training training mode
  """
  return tf.layers.dropout(inputs=input,training=training)

def cnnModel(features,labels,mode):
  """
  The cnn features
  @features the input features map (a python dictionnary - each key is the 
    name of the feature, each value is an array containing all of that
    feature's values)
  @labels contains a list of predictions for our examples
  @mode training mode or evaluation
  """

  # input
  # 28x28 pixel images - greyscale (1)
  # [batchsize,28,28,1]
  # batchsize = -1 => computed dynamically (depends on input value in features["x"])
  # batchsize is a hyperparameter
  layer_input = tf.reshape(features['x'],[-1,28,28,1])

  # first convolutional layer
  # apply 32 5x5 filters to the input layer and use a ReLU
  # output shape: [batchsize,28,28,32] (because we use same padding than input
  #                with 32 channels)
  convLayer1 = convolutionalLayer(layer_input,32,[5,5],tf.nn.relu)

  # first pooling layer
  # output shape: [batchsize,14,14,32] (because the filter reduces size by 50%
  #               - [2,2])
  poolLayer1 = poolingLayer(convLayer1,[2,2],2)

  # second convolutional layer
  # apply 64 5x5 filters and use a ReLU
  # output shape: [batchsize,14,14,64]
  convLayer2 = convolutionalLayer(poolLayer1,64,[5,5],tf.nn.relu)

  # second pooling layer
  # output shape: [batchsize,7,7,64] (reduce by 50% again)
  poolLayer2 = poolingLayer(convLayer2,[2,2],2)

  # first dense layer
  # need to flatten the previous output => shape: [batchsize,features]
  # output: [batchsize,1024]
  poolReshape = tf.reshape(poolLayer2,[-1,7*7*64])
  denseLayer1 = denseLayer(poolReshape,1024,tf.nn.relu)

  # adding dropout activation
  # dropout rate: 40% (40% of the elements will be randomly dropped out during
  #                    the training only)
  # mode == tf.estimator.ModeKeys.TRAIN => if true, apply dropout
  # output: [batchsize,1024]
  dropoutAct = dropout(denseLayer1,0.4,mode == tf.estimator.ModeKeys.TRAIN)

  # second dense layer - final output layer (logits layer)
  # use linear activation
  # output: [batchsize,10] - one for each target class 0-9
  logits = denseLayer(dropoutAct,10,None)

  # generate predictions for each class
  # axis = 1 => axis of the input tensor along which to find the greatest value
  # arg name allow to reference it later
  predictions = {
    "classes": tf.argmax(input=logits,axis=1),
    "probabilities": tf.nn.softmax(logits, name="sft_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

  # calculate loss with cross entropy
  # measure how closely the model's predictions match the target classes
  # one_hot perform one hot encoding in order to calculate cross-entropy
  #   depth = 10 => the number of target classes
  #
  onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=10)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,logits=logits)

  # configure the training
  # optimize loss value during training
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

  # add evaluation metrics
  eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(labels=labels, 
      predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

def trainAndEvalModel():
  """
  train the model and evaluate it
  """
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # return numpy array
  train_labels = np.asarray(mnist.train.labels,dtype=np.int32)
  eval_data = mnist.test.images # return numpy array
  eval_labels = np.asarray(mnist.test.labels,dtype=np.int32)

  # create the estimator
  # use cnnModel for training and evaluation
  # checkpoints will be saved in model_dir
  mnist_classifier = tf.estimator.Estimator(model_fn=cnnModel,
    model_dir="/tmp/mnist_cnn_model")

  # set up logging for predictions
  tensors_to_log = {"probabilities":"sft_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
    every_n_iter=50)

  # train the model
  # x training feature data
  # y training feature labels
  # batch_size : the model will train on minibatches of batch_size examples 
  #              at each step
  # steps train for 20 000 steps
  train_input = tf.estimator.inputs.numpy_input_fn(
    x={"x":train_data},y=train_labels,batch_size=100,num_epochs=None,shuffle=True)
  mnist_classifier.train(input_fn=train_input,steps=20000,hooks=[logging_hook])

  # evaluate the model
  eval_input = tf.estimator.inputs.numpy_input_fn(
    {"x":eval_data},y=eval_labels,num_epochs=1,shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input)
  print (eval_results)

def main(_):
  # CNN MODEL
  print ("CNN Model - Tutorial about layers - Tensorflow")
  trainAndEvalModel()

if __name__ == "__main__":
  tf.app.run() # run the tensorflow application

# result: 'accuracy': 0.9788
# time: less than 10 minutes