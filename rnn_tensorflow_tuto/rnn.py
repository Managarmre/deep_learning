# @author Pauline Houlgatte
# based on https://www.youtube.com/watch?v=BtAVBeLuigI&feature=youtu.be

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import numpy as np
import math
import time
import my_txtutils as txt # code from github of Martin Gorner
# comment recursive option line 227 if you use python < 3.5

CELLSIZE = 512
ALPHASIZE = txt.ALPHASIZE
NLAYERS = 3
SEQLEN = 30

BATCHSIZE = 100
text = "shakespeare/*txt"

def main():
	codetext, valitext, bookranges = txt.read_data_files(text, validation=False)
	dropout_pkeep = 0.8 

	print "==== tensorflow ===="

	Xd = tf.placeholder(tf.uint8,[None,None])
	X = tf.one_hot(Xd,ALPHASIZE,1.0,0.0)
	Yd = tf.placeholder(tf.uint8,[None,None])
	Y_ = tf.one_hot(Yd,ALPHASIZE,1.0,0.0)
	Hin = tf.placeholder(tf.float32,[None,CELLSIZE*NLAYERS]) # input
	pkeep = tf.placeholder(tf.float32)
	batchsize = tf.placeholder(tf.int32)

	cell = [rnn.GRUCell(CELLSIZE) for _ in range(NLAYERS)] # defines weights and biaises internally
	dropcells = [rnn.DropoutWrapper(c,input_keep_prob=pkeep) for c in cell]
	mcell = tf.nn.rnn_cell.MultiRNNCell(dropcells,state_is_tuple=False) # adding three layers
	multicell = rnn.DropoutWrapper(mcell, output_keep_prob=pkeep)  # dropout for the softmax layer


	Hr, H = tf.nn.dynamic_rnn(multicell, X, initial_state=Hin) # roll in width
	# each X is a character encoded with an alphabet (size = 98)
	# size of X : batch_size[30 * 98] (30 is the lenght of sequence and 98 for the alphabet)

	Hf = tf.reshape(Hr, [-1,CELLSIZE])
	Ylogits = layers.linear(Hf,ALPHASIZE)
	Y = tf.nn.softmax(Ylogits)
	Yp = tf.argmax(Y,1)
	Yp = tf.reshape(Yp,[batchsize,-1])

	loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
	loss = tf.reshape(loss,[batchsize,-1])
	predictions = tf.argmax(Y,1)
	predictions = tf.reshape(predictions,[batchsize,-1])
	train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	# training loop
	inH = np.zeros([BATCHSIZE,CELLSIZE*NLAYERS])


	epoch_size = len(codetext) // (BATCHSIZE * SEQLEN)
	txt.print_data_stats(len(codetext), len(valitext), epoch_size)
	seqloss = tf.reduce_mean(loss, 1)
	batchloss = tf.reduce_mean(seqloss)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(Yd, tf.cast(Yp, tf.uint8)), tf.float32))
	loss_summary = tf.summary.scalar("batch_loss", batchloss)
	acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
	summaries = tf.summary.merge([loss_summary, acc_summary])
	step = 0
	timestamp = str(math.trunc(time.time()))
	summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
	DISPLAY_FREQ = 50
	_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN
	

	for x,y_,epoch in txt.rnn_minibatch_sequencer(codetext,BATCHSIZE,SEQLEN,nb_epochs=10):
		dic = {Xd:x,Yd:y_,Hin:inH, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
		_,y,outH = sess.run([train_step,Yp,H],feed_dict =dic)

		if step % _50_BATCHES == 0:
			dic = {Xd: x, Yd: y_, Hin: inH, pkeep: 1.0, batchsize: BATCHSIZE}  # no dropout for validation
			y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=dic)
			txt.print_learning_learned_comparison(x, y, l, bookranges, bl, acc, epoch_size, step, epoch)
			summary_writer.add_summary(smm, step)

		inH = outH
		

    	step += BATCHSIZE * SEQLEN

if __name__ == '__main__':
	main()

# accuracy = 0.56