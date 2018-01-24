# @author Pauline Houlgatte
# based on https://www.youtube.com/watch?v=BtAVBeLuigI&feature=youtu.be


import tensorflow as tf
# MNIST is a database of handwritten digits available on http://yann.lecun.com/exdb/mnist/.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import math
import numpy as np

def batchnorm(Y,test,it,b,convolutional=False):
	exp_avg = tf.train.ExponentialMovingAverage(0.999, it)
	epsilon = 1e-5
	if convolutional:
		mean,variance = tf.nn.moments(Y,[0,1,2]) # give mean and variance
	else:
		mean,variance = tf.nn.moments(Y,[0])
	update = exp_avg.apply([mean,variance])
	m = tf.cond(test, lambda: exp_avg.average(mean), lambda: mean)
	v = tf.cond(test, lambda: exp_avg.average(variance), lambda: variance)
	Ybn = tf.nn.batch_normalization(Y, m, v, b, None, epsilon)
	return Ybn, update

def compatible_convolutional_noise_shape(Y):
	noiseshape = tf.shape(Y)
	noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
	return noiseshape

def main():
	print "==== tensorflow ===="

	print "INITIALIZATION"

	X = tf.placeholder(tf.float32,[None,28,28,1]) # receive the value of images during the training
	# None = the number of images
	# 28*28 = size of images (the number of pixel)
	# 1 = one data per pixel (1 is greyscale, 3 for color)

	it = tf.placeholder(tf.int32)

	K = 24
	L = 48
	M = 64
	N = 200

	pkeep = tf.placeholder(tf.float32)
	pkeep_conv = tf.placeholder(tf.float32)

	# selection for batch normalization
	test = tf.placeholder(tf.bool)

	# we have one weighted matrix and one bias for each layer
	# initialize each one with random values (truncated_normal = random)
	W1 = tf.Variable(tf.truncated_normal([6,6,1,K], stddev=0.1))
	B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
	W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
	B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
	W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
	B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

	W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
	B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
	W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
	B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

	print "MODEL WITH 5 LAYERS USING CONVOLUTIONAL NEURAL NETWORK AND BATCH NORM"

	Y1_ = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
	Y1_norm, update1 = batchnorm(Y1_,test,it,B1,convolutional=True)
	Y1 = tf.nn.relu(Y1_norm)
	Y1_dropout = tf.nn.dropout(Y1, pkeep_conv, compatible_convolutional_noise_shape(Y1))

	Y2_ = tf.nn.conv2d(Y1_dropout,W2,strides=[1,2,2,1],padding='SAME')
	Y2_norm, update2 = batchnorm(Y2_,test,it,B2,convolutional=True)
	Y2 = tf.nn.relu(Y2_norm)
	Y2_dropout = tf.nn.dropout(Y2, pkeep_conv, compatible_convolutional_noise_shape(Y2))

	Y3_ = tf.nn.conv2d(Y2_dropout,W3,strides=[1,2,2,1],padding='SAME')
	Y3_norm, update3 = batchnorm(Y3_,test,it,B3,convolutional=True)
	Y3 = tf.nn.relu(Y3_norm)
	Y3_dropout = tf.nn.dropout(Y3, pkeep_conv, compatible_convolutional_noise_shape(Y3))
	Y3_reshape = tf.reshape(Y3_dropout,shape=[-1,7*7*M])

	Y4_ = tf.matmul(Y3_reshape,W4)
	Y4_norm, update4 = batchnorm(Y4_,test,it,B4)
	Y4_relu = tf.nn.relu(Y4_norm)
	Y4 = tf.nn.dropout(Y4_relu,pkeep)

	Ylogits = tf.matmul(Y4,W5)+B5 # biases + normalized values
	Y = tf.nn.softmax(Ylogits) # we use softmax just for the last layer

	update = tf.group(update1,update2,update3,update4)

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
		train_data = {X:batch_X,Y_:batch_Y,pkeep:0.75,lr:learning_rate,test:False,pkeep_conv:1.0}

		# train loop
		sess.run(train_step,feed_dict=train_data)

		# train data
		a,c = sess.run([accuracy,cross_entropy],feed_dict=train_data)
		sess.run(update,{X:batch_X,Y_:batch_Y,test:False,it:i,pkeep:1.0,pkeep_conv:1.0})

		# test loop
		mnist_x = mnist.test.images
		mnist_x = np.reshape(mnist_x,(-1,28,28,1))
		test_data = {X:mnist_x, Y_:mnist.test.labels,pkeep:1.,lr:learning_rate,pkeep_conv:1.0,test:False}

		# test data
		# no dropout during evaluation
		a,c = sess.run([accuracy,cross_entropy],feed_dict=test_data)

		max_a = a if a > max_a else max_a

		if (i % 5000) == 0:
			print i

	print max_a*100,"% "

if __name__ == '__main__':
	main()

# accuracy => 99.6%