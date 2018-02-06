# @author Pauline Houlgatte
# version 1
# date 01/02/18

# see https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
# see https://www.tensorflow.org/get_started/premade_estimators

import myData
import tensorflow as tf

def myModel(features,labels,mode):
    """
    define my model
    @features 
    @labels
    @mode 
    """
    return 

def main(_):
    print ("Try Estimator")

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()