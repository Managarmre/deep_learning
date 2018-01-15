# @author Pauline Houlgatte
# based on https://www.youtube.com/watch?v=BtAVBeLuigI&feature=youtu.be


import tensorflow as tf
# MNIST is a database of handwritten digits available on http://yann.lecun.com/exdb/mnist/.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import math

# cross entropy -- loss function
def crossEntropy(Y_,Y):
	res = - tf.reduce_sum(Y_*tf.log(Y))
	return res

def learning(init,X,Y_,train_step,accuracy,cross_entropy,lr):
	sess = tf.Session() # run the graph / node
	sess.run(init)

	max_a = 0

	max_learning_rate = 0.003
	min_learning_rate = 0.001
	decay_speed = 2000. # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
	
	
	for i in range(10000):

		learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
		# create dict
		batch_X,batch_Y = mnist.train.next_batch(100)
		train_data = {X:batch_X,Y_:batch_Y,lr:learning_rate}

		# train loop
		sess.run(train_step,feed_dict=train_data)

		# train data
		a,c = sess.run([accuracy,cross_entropy],feed_dict=train_data)

		# test data
		test_data = {X:mnist.test.images, Y_:mnist.test.labels}
		a,c = sess.run([accuracy,cross_entropy],feed_dict=test_data)

		max_a = a if a > max_a else max_a

		if (i % 5000) == 0:
			print i

	print max_a*100,"% "

def main():
	print "==== tensorflow ===="

	print "INITIALIZATION"

	X = tf.placeholder(tf.float32,[None,28,28,1]) # receive the value of images during the training
	# None = the number of images
	# 28*28 = size of images (the number of pixel)
	# 1 = one data per pixel (1 is greyscale, 3 for color)

	K = 200
	L = 100
	M = 60
	N = 30

	# we have one weighted matrix and one bias for each layer
	# initialize each one with random values (truncated_normal = random)
	W1 = tf.Variable(tf.truncated_normal([28*28,K], stddev=0.1))
	B1 = tf.Variable(tf.zeros([K]))
	W2 = tf.Variable(tf.truncated_normal([K,L], stddev=0.1))
	B2 = tf.Variable(tf.zeros([L]))
	W3 = tf.Variable(tf.truncated_normal([L,M], stddev=0.1))
	B3 = tf.Variable(tf.zeros([M]))
	W4 = tf.Variable(tf.truncated_normal([M,N], stddev=0.1))
	B4 = tf.Variable(tf.zeros([N]))
	W5 = tf.Variable(tf.truncated_normal([N,10], stddev=0.1))
	B5 = tf.Variable(tf.zeros([10]))

	init = tf.global_variables_initializer() #tf.initialize_all_variables() is depreciated

	print "MODEL WITH 5 LAYERS USING RELU AND LEARNING RATE DECAY"

	X_reshape = tf.reshape(X,[-1,28*28]) # we need an image on a line
	Y1 = tf.nn.relu(tf.matmul(X_reshape,W1)+B1)
	# use the result of the previous layer as input data
	Y2 = tf.nn.relu(tf.matmul(Y1,W2)+B2)
	Y3 = tf.nn.relu(tf.matmul(Y2,W3)+B3)
	Y4 = tf.nn.relu(tf.matmul(Y3,W4)+B4)
	Y = tf.nn.softmax(tf.matmul(Y4,W5)+B5) # we use softmax just for the last layer

	Y_ = tf.placeholder(tf.float32,[None,10]) # to receive the label // the correct answers
	# we encode the answer on 10 elements (one per value -- 0/1/2/../9)

	print "SUCESS METRIC"
	cross_entropy = crossEntropy(Y_,Y)

	# compute the % of correct answers found
	is_correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
	accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

	print "TRAINING STEP"
	# optimization
	# we asked the optimizer to minimize the cross entropy
	# tensorflow compute the derivative of cross entropy function
	lr = tf.placeholder(tf.float32)
	optimizer = tf.train.GradientDescentOptimizer(lr) # 0.003 => 0.001 
	train_step = optimizer.minimize(cross_entropy)

	print "RUN"
	learning(init,X_reshape,Y_,train_step,accuracy,cross_entropy,lr)

if __name__ == '__main__':
	main()

# accuracy => 98%