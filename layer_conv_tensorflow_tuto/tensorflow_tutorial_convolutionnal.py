# @author Pauline Houlgatte
# based on https://www.youtube.com/watch?v=BtAVBeLuigI&feature=youtu.be


import tensorflow as tf
# MNIST is a database of handwritten digits available on http://yann.lecun.com/exdb/mnist/.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import math
import numpy as np

def main():
	print "==== tensorflow ===="

	print "INITIALIZATION"

	X = tf.placeholder(tf.float32,[None,28,28,1]) # receive the value of images during the training
	# None = the number of images
	# 28*28 = size of images (the number of pixel)
	# 1 = one data per pixel (1 is greyscale, 3 for color)

	K = 4
	L = 8
	M = 12
	N = 200

	# we have one weighted matrix and one bias for each layer
	# initialize each one with random values (truncated_normal = random)
	W1 = tf.Variable(tf.truncated_normal([5,5,1,K], stddev=0.1))
	B1 = tf.Variable(tf.ones([K])/10)
	W2 = tf.Variable(tf.truncated_normal([4,4,K,L], stddev=0.1))
	B2 = tf.Variable(tf.ones([L])/10)
	W3 = tf.Variable(tf.truncated_normal([4,4,L,M], stddev=0.1))
	B3 = tf.Variable(tf.ones([M])/10)
	W4 = tf.Variable(tf.truncated_normal([7*7*M,N], stddev=0.1))
	B4 = tf.Variable(tf.ones([N])/10)
	W5 = tf.Variable(tf.truncated_normal([N,10], stddev=0.1))
	B5 = tf.Variable(tf.zeros([10])/10)

	print "MODEL WITH 5 LAYERS USING CONVOLUTIONAL NEURAL NETWORK"

	# convolutional from layer 1 to 3
	Y1 = tf.nn.relu(tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')+B1)
	Y2 = tf.nn.relu(tf.nn.conv2d(Y1,W2,strides=[1,2,2,1],padding='SAME')+B2)
	Y3 = tf.nn.relu(tf.nn.conv2d(Y2,W3,strides=[1,2,2,1],padding='SAME')+B3)
	Y3_ = tf.reshape(Y3,shape=[-1,7*7*M]) #full connected
	Y4 = tf.nn.relu(tf.matmul(Y3_,W4)+B4)
	Ylogits = tf.matmul(Y4,W5)+B5
	Y = tf.nn.softmax(Ylogits) # we use softmax just for the last layer

	Y_ = tf.placeholder(tf.float32,[None,10]) # to receive the label // the correct answers
	# we encode the answer on 10 elements (one per value -- 0/1/2/../9)

	print "SUCESS METRIC"
	
	# cross entropy -- loss function
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
	cross_entropy = tf.reduce_mean(cross_entropy)*100

	# compute the % of correct answers found
	is_correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
	accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

	print "TRAINING STEP"
	# optimization
	# we asked the optimizer to minimize the cross entropy
	# tensorflow compute the derivative of cross entropy function
	lr = tf.placeholder(tf.float32)
	optimizer = tf.train.AdamOptimizer(lr) # 0.003 => 0.001 
	train_step = optimizer.minimize(cross_entropy)

	init = tf.global_variables_initializer() #tf.initialize_all_variables() is depreciated

	print "RUN"

	sess = tf.Session() # run the graph / node
	sess.run(init)

	max_a = 0

	max_learning_rate = 0.003
	min_learning_rate = 0.001
	decay_speed = 2000. # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
	
	
	for i in range(10000):

		learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
		# create dict (100 images with 100 labels)
		batch_X,batch_Y = mnist.train.next_batch(100)
		batch_X = np.reshape(batch_X,(-1,28,28,1))
		# dropout for training step
		train_data = {X:batch_X,Y_:batch_Y,lr:learning_rate}

		# train loop
		sess.run(train_step,feed_dict=train_data)

		# train data
		a,c = sess.run([accuracy,cross_entropy],feed_dict=train_data)

		# test loop
		mnist_x = mnist.test.images
		mnist_x = np.reshape(mnist_x,(-1,28,28,1))
		test_data = {X:mnist_x, Y_:mnist.test.labels,lr:learning_rate}

		# test data
		# no dropout during evaluation
		a,c = sess.run([accuracy,cross_entropy],feed_dict=test_data)

		max_a = a if a > max_a else max_a

		if (i % 5000) == 0:
			print i

	print max_a*100,"% "

if __name__ == '__main__':
	main()

# accuracy => 98.7% avec GradientDescentOptimizer / 99.3% avec adamOptimizer