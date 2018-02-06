# @author Pauline Houlgatte
# version 1
# date 01/02/18

# see https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
# see https://www.tensorflow.org/get_started/premade_estimators

import myData
import tensorflow as tf
import numpy as np

BATCHSIZE = 1

def main(_):
    print ("Try Premade Estimator")

    Labels = ["leopard","camera","dalmatian","human","cougar","wild cat"]

    # recover train and test data
    dataTrain = myData.loadData("Database/data.txt")
    train_x,train_y = myData.inputData(dataTrain,"Database/TrainData",6912)
    # dataTest = myData.loadData("Database/test.txt")
    # test_x,test_y = myData.inputData(dataTest,"Database/TestData",6912)

    # define the feature columns
    #   provide many options for representing data to the model
    #   describe how to use the input
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # print train_x.keys()
    # print max(train_x.keys())
    # exit()

    # print Labels
    # print train_y
    # exit()

    # print my_feature_columns
    # print len(my_feature_columns)
    # exit()

    # create three hidden layer DNN with 40 units each
    # DNNClassifier is useful for deep models that perform multiclass classification
    #   6 targets classes [0 to 5]
    classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,
        hidden_units=[1],#40,40,40],
        n_classes=6)
        # ,label_vocabulary=Labels)

    print "====> begin training"

    # train the model
    classifier.train(input_fn=lambda:myData.trainData(train_x,train_y,BATCHSIZE),
        steps=10)

    print "====> end of training"

    """
    print "====> eval result"
    eval_result = classifier.evaluate(input_fn=lambda:myData.evalTraining(
        test_x,test_y,BATCHSIZE))
    print '\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result)

    """

    print "End of program"

if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()