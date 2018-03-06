# @author Pauline Houlgatte
# date 09/02/18

# see https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
# see https://www.tensorflow.org/get_started/custom_estimators

import myData
import tensorflow as tf

BATCHSIZE = 100
STEP = 1000

def myModel(features,labels,mode,params):
    """
    define my model (same than premate estimator)
    @features input data (from input fn function)
    @labels batch labels (from input fn functions)
    @mode train / evaluate or predict mode
    @params additional configuration
    """

    # define input layer
    #   convert the feature dict into input for your model (input layer)
    net = tf.feature_column.input_layer(features,params['feature_columns'])

    # hidden layers
    #   three hidden layers (dense) with 4 units each
    #   units = number of neurons in each layer
    #   activation = activation function (here we use ReLU)
    #   net is the current layer (at the beginning this is the input layer 
    #       for example, then we take the previous layers'output as input). 
    for units in params['hidden_units']:
        net = tf.layers.dense(net,units=units,activation=tf.nn.relu)

    # output layer
    #   no activation function
    logits = tf.layers.dense(net,params['n_classes'],activation=None)

    # compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

    # compute evaluation metrics
    predicted_classes = tf.argmax(logits,1)
    accuracy = tf.metrics.accuracy(labels=labels,predictions=predicted_classes)
    metrics = {'accuracy':accuracy}
    tf.summary.scalar('accuracy',accuracy[1])

    # evaluate mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)

    # training mode
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)

def main(_):
    print ("Try Estimator")

    print "===> LOAD DATA"

    # recover train and test data
    dataTrain = myData.loadData("Database/data.txt")
    train_x,train_y = myData.inputData(dataTrain,"Database/TrainData",6912)
    dataTest = myData.loadData("Database/test.txt")
    test_x,test_y = myData.inputData(dataTest,"Database/TestData",6912)

    # define the feature columns
    #   provide many options for representing data to the model
    #   describe how to use the input
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # call your model
    classifier = tf.estimator.Estimator(model_fn=myModel,
        params={'feature_columns':my_feature_columns,'hidden_units':[4,4,4],
        'n_classes':4})

    print "===> TRAINING"
    # train your model
    classifier.train(input_fn=lambda:myData.trainData(train_x,train_y,BATCHSIZE),
        steps=STEP)

    print "===> EVALUATE"
    # evaluate your model
    eval_result = classifier.evaluate(input_fn=lambda:myData.evalTraining(test_x,test_y,BATCHSIZE))
    
    print '\n===> Test set accuracy: {accuracy:0.3f}\n'.format(**eval_result)

if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

# accuracy 40%
# execution time: around 10 minutes