# @author Pauline Houlgatte
# based on https://www.youtube.com/watch?v=BtAVBeLuigI&feature=youtu.be


import tensorflow as tf
# MNIST is a database of handwritten digits available on http://yann.lecun.com/exdb/mnist/.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# normalize / unused
def normalization(weighted_sum):
	return weighted_sum

# activation function / unused
def softmax(weighted_sum):
	return np.exp(weighted_sum) / normalisation(weighted_sum)

#
def fctY(X,W,b):
	Y = tf.nn.softmax(tf.matmul(X,W)+b)
	return Y

# cross entropy -- loss function
def crossEntropy(Y_,Y):
	res = - tf.reduce_sum(Y_*tf.log(Y))
	return res

def learning(init,X,Y_,train_step,accuracy,cross_entropy):
	sess = tf.Session() # run the graph / node
	sess.run(init)

	for i in range(10000):
		# create dict
		batch_X,batch_Y = mnist.train.next_batch(100)
		train_data = {X:batch_X,Y_:batch_Y}

		# train loop
		sess.run(train_step,feed_dict=train_data)

		# train data
		a,c = sess.run([accuracy,cross_entropy],feed_dict=train_data)

		# test data
		test_data = {X:mnist.test.images, Y_:mnist.test.labels}
		a,c = sess.run([accuracy,cross_entropy],feed_dict=test_data)

		if (i % 5000) == 0:
			print i

	print a*100,"% ",c

def main():
	print "==== tensorflow ===="

	print "INITIALIZATION"

	X = tf.placeholder(tf.float32,[None,28,28,1]) # receive the value of images during the training
	# None = the number of images
	# 28*28 = size of images (the number of pixel)
	# 1 = one data per pixel (1 is greyscale, 3 for color)

	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))

	init = tf.global_variables_initializer() #tf.initialize_all_variables() is depreciated

	print "MODEL"

	X_reshape = tf.reshape(X,[-1,784]) # we need an image on a line
	Y = fctY(X_reshape,W,b)

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
	optimizer = tf.train.GradientDescentOptimizer(0.003) # 0.003 = learning rate
	train_step = optimizer.minimize(cross_entropy)

	print "RUN"
	learning(init,X_reshape,Y_,train_step,accuracy,cross_entropy)

if __name__ == '__main__':
	main()

# accurate = 92%