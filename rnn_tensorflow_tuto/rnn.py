# @author Pauline Houlgatte
# based on https://www.youtube.com/watch?v=BtAVBeLuigI&feature=youtu.be

import tensorflow as tf

CELLSIZE = 512
ALPHASIZE = 98
NLAYERS = 3
SEQLEN = 30

def main():

	print "==== tensorflow ===="

	cell = tf.nn.rnn_cell.GRUCell(CELLSIZE) # defines weights and biaises internally
	mcell = tf.nn.rnn_cell.MultiRNNCell([cell]*NLAYERS,state_is_tuple=False) # adding three layers
	Hr, H = tf.nn.dynamic_rnn(mcell, X, initial_state=Hin) # roll in width
	# each X is a character encoded with an alphabet (size = 98)
	# size of X : batch_size[30 * 98] (30 is the lenght of sequence and 98 for the alphabet)
	Hf = tf.reshape(Hr, [-1,CELLSIZE])
	Ylogits = layers.linear(Hf,ALPHASIZE)
	Y = tf.nn.softmax(Ylogits)


if __name__ == '__main__':
	main()
