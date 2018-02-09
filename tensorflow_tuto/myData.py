# @author Pauline Houlgatte
# date 01/02/18

# see https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
# see https://www.tensorflow.org/get_started/premade_estimators

import cv2
import tensorflow as tf
import numpy as np

DATA_SIZE = 6912 # (48*48*3) for 48x48 pixel images (RGB)
BATCHSIZE = 100

def loadData(file):
    """
    load data from txt file
    @file the name of txt file (string)
    """
    f = open(file,"r")
    # print f.read()
    return f

def inputData(data,folder,size):
    """
    recover / supply data for training, evaluating and prediction
    @data input data
    @folder the folder path to recover image data
    @size size of images (width * height)
    """
    # a python dictionnary - each key is the name of the feature, 
    #   each value is an array containing all of that feature's values
    # feature = {"name":"value"}
    features = {}
    # contains a list of predictions for our examples
    # label = ["result"]
    # labelsStr = []
    labels = []

    values = [[] for i in range(size)]
    values = np.array(values)
    # recover train data from txt file
    for d in data:
        if str(d) == ";" or str(d) == ";\n":
            continue
        name,label = d.split(":")
        string = str(folder)+"/"+str(name)+".jpg"
        im = cv2.imread(string)
        value = im.reshape(1,np.size(im))
        value = value.transpose()
        values = np.hstack((values,value))
        label = label[:len(label)-1]
        if label != 'None':
            labels.append(int(label))
            # labelsStr.append(str(label))
        else:
            labels = None

    i = 1
    for v in values:
        # key must be a string
        features[str(i)] = v
        i += 1

    data.close()
    return features,labels#, labelsStr

def trainData(features,labels,batch_size):
    """
    training the agent
    @features the input features map
    @labels list of predictions
    @batch_size
    """
    # convert the input data into TFRecordDataset (tensorflow API)
    # TFRecordDataset contains methods to create and transform datasets
    # for more information about dataset
    # see https://www.tensorflow.org/programmers_guide/datasets
    dataset = tf.data.TFRecordDataset.from_tensor_slices((dict(features),labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # build the iterator and return the end of the pipeline
    return dataset.make_one_shot_iterator().get_next()

def evalTraining(features,labels,batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features,labels)

    dataset = tf.data.TFRecordDataset.from_tensor_slices(inputs)

    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset

def main(_):
    print ("Try Estimator")

    print "==== Load Train Data ===="
    dataTrain = loadData("Database/data.txt")
    featuresTrain,labelsTrain = inputData(dataTrain,"Database/TrainData",DATA_SIZE)

    print "==== Train Data ===="
    resTrain = trainData(featuresTrain,labelsTrain,BATCHSIZE)

    print "==== Load Test Data ==="
    dataTest = loadData("Database/test.txt")
    featuresTest,labelsTest = inputData(dataTest,"Database/TestData",DATA_SIZE)

    print "==== Eval Training ===="
    resEval = evalTraining(featuresTest,labelsTest,BATCHSIZE)
    # print resEval

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()