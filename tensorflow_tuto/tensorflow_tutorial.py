# @author Pauline Houlgatte
# https://www.tensorflow.org/tutorials/image_recognition


import tensorflow as tf
# MNIST is a database of handwritten digits available on http://yann.lecun.com/exdb/mnist/.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def main():
	print "==== tensorflow ===="


if __name__ == '__main__':
	main()

# accurate = 92%