# @author Pauline Houlgatte
# based on https://www.youtube.com/watch?v=BtAVBeLuigI&feature=youtu.be


import tensorflow as tf
from tensorflow.contrib import layers, learn
# MNIST is a database of handwritten digits available on http://yann.lecun.com/exdb/mnist/.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def test_data_input_fn():
	batch_X,batch_Y = mnist.train.next_batch(100)
	return tf.train.shuffle_batch([tf.constant(batch_X), tf.constant(batch_Y)],batch_size=100, capacity=1100, min_after_dequeue=1000, enqueue_many=True)

def train_data_input_fn():
	return tf.train.shuffle_batch([tf.constant(mnist.train.images), tf.constant(mnist.train.labels)],batch_size=100, capacity=1100, min_after_dequeue=1000, enqueue_many=True)

def eval_test_data_input_fn():
	batch_X,batch_Y = mnist.train.next_batch(100)
	return tf.constant(batch_X), tf.constant(batch_Y)

def eval_train_data_input_fn():
	return tf.constant(mnist.test.images), tf.constant(mnist.test.labels)

def predict_input_fn():
	return tf.constant(mnist.test.images)

def conv_model(X,Y_,mode):
	XX = tf.reshape(X,[-1,28,28,1])
	Y1 = layers.conv2d(XX,num_outputs=6,kernel_size=[6,6],padding="same") # define weight and biases automatically
	Y2 = layers.conv2d(Y1,num_outputs=12,kernel_size=[5,5],stride=2,padding="same")
	Y3 = layers.conv2d(Y2,num_outputs=24,kernel_size=[4,4],stride=2,padding="same")
	Y4 = layers.flatten(Y3)
	Y5 = layers.relu(Y4,200)
	Ylogits = layers.linear(Y5,10)
	prob = tf.nn.softmax(Ylogits)
	digi = tf.cast(tf.argmax(prob,1),tf.uint8)

	predictions = {"probabilities":prob,"digits":digi}
	YY = layers.linear(Y_,1)
	evaluations = {"accuracy":tf.metrics.accuracy(digi, YY)} 

	loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
	loss = tf.reduce_mean(loss)*100
	train = layers.optimize_loss(loss,tf.train.get_global_step(),0.003,"Adam")

	return learn.ModelFnOps(mode,predictions,loss,train,evaluations)

def main():
	print "==== tensorflow ===="

	estimator = learn.Estimator(model_fn=conv_model)
	estimator.fit(input_fn=test_data_input_fn, steps=10000)
	evaluation = estimator.evaluate(input_fn=eval_test_data_input_fn, steps=10000)
	digits = estimator.predict(input_fn=predict_input_fn)

	print " ======================= RESULT ======================= "
	print evaluation["accuracy"]*100

	estimator.fit(input_fn=train_data_input_fn, steps=10000)
	evaluation = estimator.evaluate(input_fn=eval_train_data_input_fn, steps=10000)

	print " ======================= RESULT ======================= "
	print evaluation["accuracy"]*100


	# for predict in digits:
	# 	predicted_class = predict['digits']
	# 	probability = predict['probabilities']
	# 	print (predicted_class, probability)

if __name__ == '__main__':
	main()