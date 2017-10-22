import pandas as pd
import data_process
import numpy as np
import fst
from keras.datasets import imdb
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Input, Dense, Masking, Merge
from keras.layers import LSTM, GRU
from keras.layers.core import Reshape
from keras.layers.noise import GaussianNoise
from keras.layers import Dropout, Flatten, Conv1D, Conv2D, MaxPool1D, MaxPool2D, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.engine.training import Model
from keras.preprocessing import sequence
from keras.callbacks import LearningRateScheduler,EarlyStopping,TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from scipy.sparse import csr_matrix
import tensorflow as tf
import keras
from keras.constraints import maxnorm
from keras.regularizers import l2
import seq2seq
from seq2seq.models import AttentionSeq2Seq, Seq2Seq
import time
from config import Config as cfg
from scipy import sparse
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils.np_utils import to_categorical
import math
import keras.backend as K
import tensorflow as tf
from matplotlib.pyplot import axis
from bokeh.server.protocol.messages import index
from bokeh.util.session_id import random
from keras.layers.wrappers import Bidirectional

from sklearn.cross_validation import train_test_split
cfg.init()


def submission(flag_y, df_test, file = '../data/submission.csv'):
	sub = pd.DataFrame()
	sub['id'] = df_test['id']
	print "please implement!"
	print "Submission samples:%d, file:%s"%(len(df_test), file)
	sub.to_csv(file, index=False)

# def experiment_sparse():
# 
# 	#Setup sparse input
# 	x_d = np.array([0, 7, 2, 3], dtype=np.int64)
# 	x_r = np.array([0, 2, 2, 3], dtype=np.int64)
# 	x_c = np.array([4, 3, 2, 3], dtype=np.int64)
# 	
# 	y = np.array([0, 1, 2, 3], dtype=np.int64)
# 	
# 	x_in = csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
# 	
# 	#Setup Model
# 	numeric_columns = 5
# 	
# 	sparse = Input(batch_shape=(None, numeric_columns,), tensor=tf.sparse_placeholder(tf.float32))
# 	
# 	predictions = Dense(1, activation='sigmoid')(sparse)
# 	
# 	optimizer = keras.optimizers.Adam()
# 	model = Model(input=[sparse], output=predictions)
# 	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# 	
# 	#Run it!
# 	model.fit(x_in, y)

def classify_generator_frag(X, y, batch_size=128, shuffle=True):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))
		X_batch = X[batch_index,:]
		y_batch = y[batch_index,:]
		
		x_l_batch = X_batch[:,0:cfg.max_left_input_len]
		x_m_batch = X_batch[:,cfg.max_left_input_len:cfg.max_mid_input_len + cfg.max_left_input_len]
		x_r_batch = X_batch[:,cfg.max_mid_input_len + cfg.max_left_input_len:]
		#reverse the right input
		x_r_batch = x_r_batch[:, -1::-1]
		# reshape X to be [samples, time steps, features]
# 		for i, j in enumerate(batch_index):
# 			tmpx = X[j]
# 			for t in range(X.shape[1]):
#  				X_batch[i,t,tmpx[t]] = 1.0

		counter += 1
		yield [x_l_batch, x_m_batch, x_r_batch], y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0

def classify_generator(X, y, batch_size=128, shuffle=True):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))
		X_batch = X[batch_index,:]
		y_batch = y[batch_index,:]
		# reshape X to be [samples, time steps, features]
# 		for i, j in enumerate(batch_index):
# 			tmpx = X[j]
# 			for t in range(X.shape[1]):
#  				X_batch[i,t,tmpx[t]] = 1.0

		counter += 1
		yield X_batch, y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0
			
def classify_generator_extend(X, y, batch_size=128, shuffle=True):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))
		X_batch = X[batch_index,:]
		y_batch = y[batch_index,:]
		X_batch_c = X_batch[:, 0:cfg.max_input_len]
		X_batch_e = X_batch[:, cfg.max_input_len:]
		counter += 1
		yield [X_batch_c, X_batch_e], y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0


def classify_generator_onehot(X, y, batch_size=128, shuffle=True):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))
# 		X_batch = X[batch_index,:]
		y_batch = y[batch_index,:]
# 		reshape X to be [samples, time steps, features]
 		for i, j in enumerate(batch_index):
 			tmpx = X[j]
 			for t in range(X.shape[1]):
  				X_batch[i,t,tmpx[t]] = 1.0

		counter += 1
		yield X_batch, y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0
			
def reshape_classify_onehot(X):
	X_out = np.zeros((X.shape[0], X.shape[1], cfg.input_classify_vocab_size))
	for i in range(X.shape[0]):
		for t in range(X.shape[1]):
			X_out[i,t,X[i,t]] = 1.0
	return X_out

def sparse_generator(X, y, batch_size=128, shuffle=True):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_vocab_size))
		y_batch = np.zeros((len(batch_index), y.shape[1], cfg.output_vocab_size))
		# reshape X to be [samples, time steps, features]
		for i, j in enumerate(batch_index):
# 			X_batch[i,:,:] = np.transpose(X[j].toarray())
			tmpx = X[j].toarray()
			for t in range(tmpx.shape[1]):
 				X_batch[i,t,tmpx[0,t]] = 1.0
			for t in range(y.shape[1]):
				y_batch[i,t,y[j, t]] = 1.0
		# reshape y to be [samples, time steps, features]	
# 		X_decode_batch = np.reshape(X_decode_batch, (X_decode_batch.shape[0], X_decode_batch.shape[1], 1))	
		counter += 1
		yield X_batch, y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0
			
def generator_teaching(X, y, batch_size=128, shuffle=True):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
		X_batch = X[batch_index]
		X_decode_batch = y[batch_index]
		y_batch = np.zeros((len(batch_index), y.shape[1], y.shape[2]))
		# reshape X to be [samples, time steps, features]
		for i, j in enumerate(batch_index):
			for t in range(y.shape[1]):
				if t > 0:
					y_batch[i,t - 1] = y[j, t]
		# reshape y to be [samples, time steps, features]	
# 		X_decode_batch = np.reshape(X_decode_batch, (X_decode_batch.shape[0], X_decode_batch.shape[1], 1))	
		counter += 1
		yield [X_batch, X_decode_batch], y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0
			
def sparse_generator_two(X, y, batch_size=128, shuffle=True):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_vocab_size))
		X_decode_batch = np.zeros((len(batch_index), y.shape[1], cfg.output_vocab_size))
		y_batch = np.zeros((len(batch_index), y.shape[1], cfg.output_vocab_size))
		# reshape X to be [samples, time steps, features]
		for i, j in enumerate(batch_index):
# 			X_batch[i,:,:] = np.transpose(X[j].toarray())
			tmpx = X[j].toarray()
			for t in range(tmpx.shape[1]):
 				X_batch[i,t,tmpx[0,t]] = 1.0
			for t in range(y.shape[1]):
				X_decode_batch[i,t,y[j, t]] = 1.0
				if t > 0:
					y_batch[i,t - 1,y[j, t]] = 1.0
		# reshape y to be [samples, time steps, features]	
# 		X_decode_batch = np.reshape(X_decode_batch, (X_decode_batch.shape[0], X_decode_batch.shape[1], 1))	
		counter += 1
		yield [X_batch, X_decode_batch], y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0
					
def generator_teaching_onehot(X, y, batch_size=128, shuffle=True):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
		X_batch = np.zeros((batch_size, X.shape[1], cfg.input_vocab_size))
		X_decode_batch = np.zeros((batch_size, y.shape[1], cfg.output_vocab_size))
		y_batch = np.zeros((batch_size, y.shape[1], cfg.output_vocab_size))
		# reshape X to be [samples, time steps, features]
		for i, j in enumerate(batch_index):
# 			X_batch[i,:,:] = np.transpose(X[j].toarray())
			for t in range(X.shape[1]):
 				X_batch[i,t,X[j, t]] = 1.0
			for t in range(y.shape[1]):
				X_decode_batch[i,t,y[j, t]] = 1.0
				if t > 0:
					y_batch[i,t - 1,y[j, t]] = 1.0
		# reshape y to be [samples, time steps, features]	
# 		X_decode_batch = np.reshape(X_decode_batch, (X_decode_batch.shape[0], X_decode_batch.shape[1], 1))	
		counter += 1
		yield [X_batch, X_decode_batch], y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0
			
			
def reshape_data(X, y):
	X_out = np.zeros((X.shape[0], X.shape[1], cfg.input_vocab_size))
	for i in range(X.shape[0]):
		for t in range(X.shape[1]):
			X_out[i,t,X[i,t]] = 1.0
	
	y_out = np.zeros((y.shape[0], y.shape[1], cfg.output_vocab_size))
	for i in range(y.shape[0]):
		for t in range(y.shape[1]):
			y_out[i,t,y[i,t]] = 1.0

					
	return X_out, y_out

def reshape_data_classify_frag(X):
# 	X_out = np.zeros((X.shape[0], X.shape[1], cfg.input_classify_vocab_size))
# 	for i in range(X.shape[0]):
# 		for t in range(X.shape[1]):
# 			X_out[i,t,X[i,t]] = 1.0
	x_l = X[:,0:cfg.max_left_input_len]
	x_m = X[:,cfg.max_left_input_len:cfg.max_mid_input_len + cfg.max_left_input_len]
	x_r = X[:,cfg.max_mid_input_len + cfg.max_left_input_len:]
	#reverse the right input
	x_r = x_r[:, -1::-1]
	return x_l, x_m, x_r


			
def reshape_data_teaching(X, y):
	X_out = X
	y_out = np.zeros((y.shape[0], y.shape[1]))
	X_d_out = y
	for i in range(y.shape[0]):
		for t in range(y.shape[1]):
			if t > 0:
				y_out[i,t - 1] = y[i, t]
					
	return X_out, X_d_out, y_out

def reshape_data_teaching_onehot(X, y):
	X_out = np.zeros((X.shape[0], X.shape[1], cfg.input_vocab_size))
	for i in range(X.shape[0]):
		for t in range(X.shape[1]):
			X_out[i,t,X[i,t]] = 1.0
	
	y_out = np.zeros((y.shape[0], y.shape[1], cfg.output_vocab_size))
	X_d_out = np.zeros((y.shape[0], y.shape[1], cfg.output_vocab_size))
	for i in range(y.shape[0]):
		for t in range(y.shape[1]):
			X_d_out[i,t,y[i,t]] = 1.0
			if t > 0:
				y_out[i,t - 1,y[i, t]] = 1.0
					
	return X_out, X_d_out, y_out

def train_teaching(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, batch_size=128, nb_epoch = 100):
	path = '../data/'
	# reshape X to be [samples, time steps, features]
# 	X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
# 	Y_valid = np.reshape(Y_valid, (Y_valid.shape[0], Y_valid.shape[1], 1))
 	X_valid, X_d_vaild, Y_valid = reshape_data_teaching(X_valid, Y_valid)
# 	print X_train.getnnz()
# 	print X_train.shape[0]
# 	print X_train.shape[1]

	board = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head)
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
#  	samples_per_epoch = batch_size
	model.fit_generator(generator=generator_teaching(X_train, Y_train, batch_size, False), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
			    	    validation_data=([X_valid, X_d_vaild], Y_valid),
			    		callbacks=[board, checkpointer])
	print 'Training time', time.time() - start_time
	# evaluate network
# 	decode_sequence(X_valid, Y_valid, extend_args)
	
	score = model.evaluate(X_valid, Y_valid, batch_size)
	p_y = model.predict(X_valid, batch_size, verbose=0)
# 	np.savez(path + ret_file_head + '.npz', p_y = p_y)
	out_list = get_predict_list(p_y)
	out_list = format_out_list(out_list)
# 	output_to_csv(out_list, df_valid, path + ret_file_head + '.csv')
	print out_list
# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)

def train_teaching_onehot(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, batch_size=128, nb_epoch = 100):

	# reshape X to be [samples, time steps, features]
# 	X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
# 	Y_valid = np.reshape(Y_valid, (Y_valid.shape[0], Y_valid.shape[1], 1))
	X_valid, X_d_vaild, Y_valid = reshape_data_teaching_onehot(X_valid, Y_valid)
# 	print X_train.getnnz()
# 	print X_train.shape[0]
# 	print X_train.shape[1]

	board = TensorBoard(log_dir='../logs/nt1/', histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head)
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
#  	samples_per_epoch = batch_size
	model.fit_generator(generator=generator_teaching_onehot(X_train, Y_train, batch_size, False), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
			    	    validation_data=([X_valid, X_d_vaild], Y_valid),
			    		callbacks=[board, checkpointer])
	print 'Training time', time.time() - start_time
	# evaluate network
# 	decode_sequence(X_valid, Y_valid, extend_args)
	
	score = model.evaluate([X_valid, X_d_vaild], Y_valid, batch_size)
	p_y = model.predict([X_valid, X_d_vaild], batch_size, verbose=0)
	out_list = get_predict_list(p_y)
	out_list = format_out_list(out_list)
# 	output_to_csv(out_list, df_valid, path + ret_file_head + '.csv')
	print out_list
# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)

def decode_sequence(input_sequence, weight_path):
	
	latent_dim = cfg.input_vocab_size
	num_encoder_tokens = cfg.input_vocab_size
	num_decoder_tokens = cfg.output_vocab_size
	
	encoder_inputs = Input(shape=(None, num_encoder_tokens))
	encoder = LSTM(cfg.input_vocab_size, return_state=True)
	_, state_h, state_c = encoder(encoder_inputs)
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]
	
	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(None, num_decoder_tokens))
	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the
	# return states in the training model, but we will use them in inference.
	decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# 	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(num_decoder_tokens, activation='softmax')
	
	encoder_model = Model(encoder_inputs, encoder_states)
	decoder_state_input_h = Input(shape=(latent_dim,))
	decoder_state_input_c = Input(shape=(latent_dim,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(
	    decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model(
	    [decoder_inputs] + decoder_states_inputs,
	    [decoder_outputs] + decoder_states)
	
	# Encode the input as state vectors.
	states_value = encoder_model.predict(input_sequence)
	
	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1, 1, num_decoder_tokens))
	# Populate the first character of target sequence with the start character.
	target_seq[0, 0, cfg.dic_output_word2i['<GO>']] = 1.
	
	# Sampling loop for a batch of sequences
	# (to simplify, here we assume a batch of size 1).
	stop_condition = False
	decoded_sentence = ''
	while not stop_condition:
		output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
		
		# Sample a token
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = cfg.dic_output_i2word[sampled_token_index]
		decoded_sentence += sampled_char
		
		# Exit condition: either hit max length
		# or find stop character.
		if (sampled_char == cfg.end_flg or len(decoded_sentence) > cfg.max_output_len):
			stop_condition = True
		
		# Update the target sequence (of length 1).
		target_seq = np.zeros((1, 1, num_decoder_tokens))
		target_seq[0, 0, sampled_token_index] = 1.
		
		# Update states
		states_value = [h, c]
	
	return decoded_sentence

def train_sparse(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, df_valid, batch_size=128, nb_epoch = 3):
	
	
	print X_valid
	print Y_valid
	path = '../data/'
	# reshape X to be [samples, time steps, features]
# 	X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
# 	Y_valid = np.reshape(Y_valid, (Y_valid.shape[0], Y_valid.shape[1], 1))
	X_valid, Y_valid = reshape_data(X_valid, Y_valid)
# 	print X_train.getnnz()
# 	print X_train.shape[0]
# 	print X_train.shape[1]

	board = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s%s_%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head,"1","1")
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
 	
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
# 	samples_per_epoch = batch_size
	model.fit_generator(generator=sparse_generator(X_train, Y_train, batch_size, False), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
	                    validation_data = (X_valid, Y_valid),
# 			    	    validation_data=sparse_generator(X_valid, Y_valid, batch_size, False), 
# 			    	    nb_val_samples=int(math.ceil(X_valid.shape[0] / float(batch_size))),
			    	    callbacks=[board, checkpointer]
			    		)
	print 'Training time', time.time() - start_time
	# evaluate network
	score = model.evaluate(X_valid, Y_valid, batch_size)
	p_y = model.predict(X_valid, batch_size, verbose=0)
	np.savez(path + ret_file_head + ".npz", p_y = p_y)
	out_list = get_predict_list(p_y)
	out_list = format_out_list(out_list)
	output_to_csv(out_list, df_valid, path + ret_file_head + ".csv")
	print out_list

# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)
def train_classify_frag(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, batch_size=128, nb_epoch = 3):
	
	
	path = '../data/'
	# reshape X to be [samples, time steps, features]
# 	X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
# 	Y_valid = np.reshape(Y_valid, (Y_valid.shape[0], Y_valid.shape[1], 1))
	x_l_valid, x_m_valid, x_r_valid = reshape_data_classify_frag(X_valid)
# 	print X_train.getnnz()
# 	print X_train.shape[0]
# 	print X_train.shape[1]

	board = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s%s_%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head,"1","1")
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
 	
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
# 	samples_per_epoch = batch_size * 2
	model.fit_generator(generator=classify_generator_frag(X_train, Y_train, batch_size, True), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
	                    validation_data = ([x_l_valid, x_m_valid, x_r_valid], Y_valid),
# 			    	    validation_data=sparse_generator(X_valid, Y_valid, batch_size, False), 
# 			    	    nb_val_samples=int(math.ceil(X_valid.shape[0] / float(batch_size))),
			    	    callbacks=[board, checkpointer]
			    		)
	print 'Training time', time.time() - start_time
	# evaluate network
	score = model.evaluate(X_valid, Y_valid, batch_size)

# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)
	
def train_classify(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, batch_size=128, nb_epoch = 3):
	
	# reshape X to be [samples, time steps, features]
# 	X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
# 	Y_valid = np.reshape(Y_valid, (Y_valid.shape[0], Y_valid.shape[1], 1))
# 	X_valid = reshape_data_classify(X_valid)
# 	print X_train.getnnz()
# 	print X_train.shape[0]
# 	print X_train.shape[1]

	board = TensorBoard(log_dir='../logs/c4/', histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head)
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
 	
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
# 	samples_per_epoch = batch_size * 2
	model.fit_generator(generator=classify_generator(X_train, Y_train, batch_size, True), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
	                    validation_data = (X_valid, Y_valid),
# 			    	    validation_data=sparse_generator(X_valid, Y_valid, batch_size, False), 
# 			    	    nb_val_samples=int(math.ceil(X_valid.shape[0] / float(batch_size))),
			    	    callbacks=[board, checkpointer]
			    		)
	print 'Training time', time.time() - start_time
	# evaluate network
	score = model.evaluate(X_valid, Y_valid, batch_size)

# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)
	
def train_classify_extend(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, batch_size=128, nb_epoch = 3):
	
	X_valid_c = X_valid[:, 0:cfg.max_input_len]
	X_valid_e = X_valid[:, cfg.max_input_len:]
	
	board = TensorBoard(log_dir='../logs/c3/', histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head)
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
 	
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
# 	samples_per_epoch = batch_size * 2
	model.fit_generator(generator=classify_generator_extend(X_train, Y_train, batch_size, True), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
	                    validation_data = ([X_valid_c, X_valid_e], Y_valid),
# 			    	    validation_data=sparse_generator(X_valid, Y_valid, batch_size, False), 
# 			    	    nb_val_samples=int(math.ceil(X_valid.shape[0] / float(batch_size))),
			    	    callbacks=[board, checkpointer]
			    		)
	print 'Training time', time.time() - start_time
	# evaluate network
	score = model.evaluate([X_valid_c, X_valid_e], Y_valid, batch_size)

# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)
	
def train_classify_onehot(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, batch_size=128, nb_epoch = 3):
	
	
	path = '../data/'
	# reshape X to be [samples, time steps, features]
	X_valid = reshape_classify_onehot(X_valid)

	board = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s%s_%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head,"1","1")
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
 	
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
# 	samples_per_epoch = batch_size * 2
	model.fit_generator(generator=classify_generator_onehot(X_train, Y_train, batch_size, True), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
	                    validation_data = (X_valid, Y_valid),
# 			    	    validation_data=sparse_generator(X_valid, Y_valid, batch_size, False), 
# 			    	    nb_val_samples=int(math.ceil(X_valid.shape[0] / float(batch_size))),
			    	    callbacks=[board, checkpointer]
			    		)
	print 'Training time', time.time() - start_time
	# evaluate network
	score = model.evaluate(X_valid, Y_valid, batch_size)

# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)
	
	
# 	return cnt / y_true.shape[0]
# 	return cnt
def arr_to_sen(arr):
	out = []
	for j in range(arr.shape[0]):
		ind = arr[j]
		if ind != cfg.pad_flg_index and ind != cfg.start_flg_index and ind != cfg.end_flg_index:
			out.append(cfg.dic_output_i2word[ind])
	return out

def arrs_to_sens(arrs):
	out_list = []
	for i in range(arrs.shape[0]):
		out_list.append(arr_to_sen(arrs[i]))
	return out_list

def format_out_list(sens):
	outs = map(lambda s: " ".join(s), sens)
	return outs

def output_to_csv(after_list, df, file="../data/test_ret.csv"):
	
# 	print after_list[0]
# 	print after_list[1]
	
	df["index"] = range(len(df))
	df = df.set_index(["index"])
	
	se = pd.Series(after_list)
	if 'after' in df.columns:
		df_ret = pd.DataFrame(columns=['id', 'before', 'after_truth', 'after'])
		df_ret['after'] = se
		df_ret['after_truth'] = df['after']
		df_corrected = df_ret[df_ret['after']==df_ret['after_truth']]
		acc = len(df_corrected)/float(len(df_ret))
		print "The corrected num is %d, real acc:%f"%(len(df_corrected), acc)
	else:
		df_ret = pd.DataFrame(columns=['id', 'before', 'after'])
		df_ret['after'] = se
		print "No ground truth data, cannot calculate real acc!"
	df_ret['id'] = df['id']
	df_ret['before'] = df['before']
	
	def copy(before, after):
		if after.strip() == cfg.unk_flg:
			return before
		else:
			return after
	df_ret['after'] = df_ret.apply(lambda x: copy(str(x['before']), str(x['after'])), axis=1)
	
	df_ret.to_csv(file, index=False)
	print 'Saved result in file:%s'%(file)
	
def get_predict_list(predict_y):
	out_list = []
	predict_y = np.argmax(predict_y, axis=2)
	out_list = arrs_to_sens(predict_y)
# 	for i in range(predict_y.shape[0]):
# 		out = []
# 		for j in range(predict_y.shape[1]):
# 			ind = predict_y[i, j]
# 			
# 			if ind != cfg.pad_flg_index:
# 				out.append(cfg.dic_output_i2word[ind])
# 		out_list.append(out)
	return out_list


	
	

def gen_model_2(depth=4, dropout=0.3):
	model = AttentionSeq2Seq(input_dim=cfg.input_vocab_size, 
							input_length=cfg.max_input_len, 
							hidden_dim=cfg.input_hidden_dim, 
							output_length=cfg.max_output_len, 
							output_dim=cfg.output_vocab_size, 
							dropout = dropout,
							depth = depth)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def gen_model_3(depth=3, dropout=0.3, peek=True, teacher_force=False):
	model = Seq2Seq(batch_input_shape=(None, cfg.max_input_len, cfg.input_vocab_size),
						 hidden_dim=cfg.input_hidden_dim, 
					 	output_length=cfg.max_output_len,
					  	output_dim=cfg.output_vocab_size, 
					  	depth=depth, teacher_force=teacher_force, dropout=dropout,
					 	peek=peek)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
def gen_model_4():
	encoder_inputs = Input(shape=(None, cfg.input_vocab_size))
	inp_norm = BatchNormalization()(encoder_inputs)
	conv_1 = Conv2D(cfg.conv_depth, (cfg.kernel_size, cfg.kernel_size), 
		padding='same', 
		kernel_initializer='he_uniform', 
		kernel_regularizer=l2(cfg.l2_lambda), 
		activation='relu')(inp_norm)
		
	conv_1 = BatchNormalization()(conv_1)
	conv_2 = Conv2D(cfg.conv_depth, (cfg.kernel_size, cfg.kernel_size), 
		padding='same', 
		kernel_initializer='he_uniform', 
		kernel_regularizer=l2(cfg.l2_lambda), 
		activation='relu')(conv_1)
	conv_2 = BatchNormalization()(conv_2)
	pool_1 = MaxPool2D(pool_size=(cfg.pool_size, cfg.pool_size))(conv_2)
	drop_1 = Dropout(cfg.drop_prob_1)(pool_1)
	flat = Flatten()(drop_1)
	hidden = Dense(cfg.hidden_size, 
		kernel_initializer='he_uniform', 
		kernel_regularizer=l2(cfg.l2_lambda), 
		activation='relu')(flat) # Hidden ReLU layer
	hidden = BatchNormalization()(hidden)
	drop = Dropout(cfg.drop_prob_2)(hidden)
	out = Dense(cfg, 
		kernel_initializer='glorot_uniform', 
		kernel_regularizer=l2(cfg.l2_lambda), 
		activation='softmax')(drop) # Output softmax layer	
def gen_model_teaching_0():
# 	embedding_len = cfg.input_vocab_size / 2
# 	embedding_layer_en = Embedding(input_dim = cfg.input_vocab_size,  
#                             output_dim = embedding_len,  
#                             input_length=cfg.max_input_len,
#                             embeddings_initializer = 'glorot_uniform',
# #                             weights=[embedding_matrix],  
#                             trainable=True)
# 	
# 	embedding_layer_de = Embedding(input_dim = cfg.output_vocab_size,  
#                             output_dim = embedding_len,  
#                             input_length=cfg.max_output_len,
#                             embeddings_initializer = 'glorot_uniform',
# #                             weights=[embedding_matrix],  
#                             trainable=True)
	# Define an input sequence and process it.
 	encoder_inputs = Input(shape=(cfg.max_input_len, cfg.input_vocab_size))
#  	input_vec_en = embedding_layer_en(encoder_inputs)
#  	reshaper_en = Reshape((cfg.max_input_len, embedding_len))
#  	input_vec_en = reshaper_en(input_vec_en)
 	
	encoder = LSTM(cfg.input_vocab_size, return_state=True)
	_, state_h, state_c = encoder(encoder_inputs)
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]
	
	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(cfg.max_output_len, cfg.output_vocab_size))
	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the
	# return states in the training model, but we will use them in inference.
# 	input_vec_de = embedding_layer_de(decoder_inputs)
# 	reshaper_de = Reshape((cfg.max_output_len, embedding_len))
# 	input_vec_de = reshaper_de(input_vec_de)
	
	decoder_lstm = LSTM(cfg.input_vocab_size, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(cfg.output_vocab_size, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	
	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
# 	extend_args = {'encoder_inputs':encoder_inputs, "encoder_states":encoder_states, "decoder_inputs":decoder_inputs,
# 				   'decoder_lstm':decoder_lstm, 'decoder_dense':decoder_dense }
	return model

def gen_model_class_0():
	
	embedding_len = cfg.input_vocab_size / 2
	
	embedding_layer = Embedding(input_dim = cfg.input_vocab_size,  
                            output_dim = embedding_len,  
                            input_length=cfg.max_input_len,
                            embeddings_initializer = 'glorot_uniform',
#                             weights=[embedding_matrix],  
                            trainable=True)
	model1 = Sequential()
	model1.add(embedding_layer)
# 	model1.add(GaussianNoise(1))
	model1.add(Conv1D(filters = 16, kernel_size = 2, ))
	model1.add(PReLU())
	model1.add(BatchNormalization())
	model1.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', ))
	model1.add(BatchNormalization())
	model1.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', kernel_regularizer=l2(0.01)))
	model1.add(BatchNormalization())
	model1.add(MaxPool1D(2))
# 	model1.add(Dropout(0.1))
	model1.add(Flatten())
	
	
	model2 = Sequential()
	model2.add(embedding_layer)
# 	model2.add(GaussianNoise(1))
	model2.add(Conv1D(filters = 16, kernel_size = 3, ))
	model2.add(PReLU())
	model2.add(BatchNormalization())
	model2.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', ))
	model2.add(BatchNormalization())
	model2.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', kernel_regularizer=l2(0.01)))
	model2.add(BatchNormalization())
	model2.add(MaxPool1D(2))
	model2.add(Flatten())
	
	model3 = Sequential()
	model3.add(embedding_layer)
# 	model3.add(GaussianNoise(1))
	model3.add(Conv1D(filters = 16, kernel_size = 5, ))
	model3.add(PReLU())
	model3.add(BatchNormalization())
	model3.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', ))
	model3.add(BatchNormalization())
	model3.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', kernel_regularizer=l2(0.01)))
	model3.add(BatchNormalization())
	model3.add(MaxPool1D(2))
	model3.add(Flatten())
	
	model4 = Sequential()
	model4.add(embedding_layer)
# 	model4.add(GaussianNoise(1))
	model4.add(Conv1D(filters = 16, kernel_size = 7, ))
	model4.add(PReLU())
	model4.add(BatchNormalization())
	model4.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', ))
	model4.add(BatchNormalization())
	model4.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', kernel_regularizer=l2(0.01)))
	model4.add(BatchNormalization())
	model4.add(MaxPool1D(2))
	model4.add(Flatten())
	
	model5 = Sequential()
	model5.add(embedding_layer)
# 	model5.add(GaussianNoise(1))
	model5.add(Conv1D(filters = 16, kernel_size = 13, ))
	model5.add(PReLU())
	model5.add(BatchNormalization())
	model5.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', ))
	model5.add(BatchNormalization())
	model5.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', kernel_regularizer=l2(0.01)))
	model5.add(BatchNormalization())
	model5.add(MaxPool1D(2))
	model5.add(Flatten())
	
	
	model6 = Sequential()
	model6.add(embedding_layer)
# 	model6.add(GaussianNoise(1))
	model6.add(Conv1D(filters = 16, kernel_size = 2, ))
	model6.add(PReLU())
	model6.add(BatchNormalization())
	model6.add(Conv1D(filters = 32, kernel_size = 3, activation='relu', ))
	model6.add(BatchNormalization())
	model6.add(Conv1D(filters = 32, kernel_size = 3, activation='relu', kernel_regularizer=l2(0.01)))
	model6.add(BatchNormalization())
	model6.add(MaxPool1D(2))
# 	model6.add(Dropout(0.1))
	model6.add(Flatten())
	
	model7 = Sequential()
	model7.add(embedding_layer)
# 	model7.add(GaussianNoise(1))
	model7.add(Conv1D(filters = 16, kernel_size = 3, ))
	model7.add(PReLU())
	model7.add(BatchNormalization())
	model7.add(Conv1D(filters = 32, kernel_size = 3, activation='relu', ))
	model7.add(BatchNormalization())
	model7.add(Conv1D(filters = 32, kernel_size = 3, activation='relu', kernel_regularizer=l2(0.01)))
	model7.add(BatchNormalization())
	model7.add(MaxPool1D(2))
# 	model7.add(Dropout(0.1))
	model7.add(Flatten())
	
	model8 = Sequential()
	model8.add(embedding_layer)
# 	model8.add(GaussianNoise(1))
	model8.add(Conv1D(filters = 16, kernel_size = 5,))
	model8.add(PReLU())
	model8.add(BatchNormalization())
	model8.add(Conv1D(filters = 32, kernel_size = 3, activation='relu', ))
	model8.add(BatchNormalization())
	model8.add(Conv1D(filters = 32, kernel_size = 3, activation='relu', kernel_regularizer=l2(0.01)))
	model8.add(BatchNormalization())
	model8.add(MaxPool1D(2))
# 	model8.add(Dropout(0.1))
	model8.add(Flatten())
	
	model9 = Sequential()
	model9.add(embedding_layer)
# 	model9.add(GaussianNoise(1))
	model9.add(Conv1D(filters = 16, kernel_size = 7, ))
	model9.add(PReLU())
	model9.add(BatchNormalization())
	model9.add(Conv1D(filters = 32, kernel_size = 3, activation='relu', ))
	model9.add(BatchNormalization())
	model9.add(Conv1D(filters = 32, kernel_size = 3, activation='relu', kernel_regularizer=l2(0.01)))
	model9.add(BatchNormalization())
	model9.add(MaxPool1D(2))
# 	model9.add(Dropout(0.1))
	model9.add(Flatten())
	
	model10 = Sequential()
	model10.add(embedding_layer)
# 	model10.add(GaussianNoise(1))
	model10.add(Conv1D(filters = 16, kernel_size = 13, ))
	model10.add(PReLU())
	model10.add(BatchNormalization())
	model10.add(Conv1D(filters = 32, kernel_size = 3, activation='relu', ))
	model10.add(BatchNormalization())
	model10.add(Conv1D(filters = 32, kernel_size = 3, activation='relu', kernel_regularizer=l2(0.01)))
	model10.add(BatchNormalization())
	model10.add(MaxPool1D(2))
# 	model10.add(Dropout(0.1))
	model10.add(Flatten())
	
	
	merged = Merge([model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, ], mode='concat')
	model = Sequential()
	model.add(merged)
# 	model1.add(DropconnectDense(units=512, rate=0.15, activation='relu'))
# 	model1.add(DropconnectDense(units=1024, rate=0.25, activation='sigmoid'))
	
	model.add(Dense(128, activation='relu', ))
# 	model.add(Dropout(0.15))
	model.add(Dense(256, activation='sigmoid', kernel_regularizer=l2(0.01), kernel_constraint=maxnorm(2.)))
	model.add(Dropout(0.25))
	
	model.add(Dense(3, activation='softmax'))
	
# 	model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
	model.compile(loss='categorical_crossentropy', optimizer = 'Adagrad', metrics=["accuracy"])
	return model
	
def gen_model_class_1():
	model_l = Sequential()
	
	embedding_layer_l = Embedding(input_dim = cfg.input_classify_vocab_size,  
                            output_dim = cfg.dim_left_embedding,  
                            input_length=cfg.max_left_input_len,
                            embeddings_initializer = 'glorot_uniform',
#                             weights=[embedding_matrix],  
                            trainable=True)
	
	embedding_layer_m = Embedding(input_dim = cfg.input_classify_vocab_size,  
                            output_dim = cfg.dim_mid_embedding,  
                            input_length=cfg.max_mid_input_len,
                            embeddings_initializer = 'glorot_uniform',
#                             weights=[embedding_matrix],  
                            trainable=True)
	
	embedding_layer_r = Embedding(input_dim = cfg.input_classify_vocab_size,  
                            output_dim = cfg.dim_right_embedding,  
                            input_length=cfg.max_right_input_len,
                            embeddings_initializer = 'glorot_uniform',
#                             weights=[embedding_matrix],  
                            trainable=True) 
	
	
	model_l.add(embedding_layer_l)
	model_l.add(GRU(cfg.input_classify_vocab_size, dropout=0.1, implementation = 2, 
						input_shape=(cfg.max_left_input_len, cfg.dim_left_embedding)))
	
	
	model_m = Sequential()
	model_m.add(embedding_layer_m)
	model_m.add(GaussianNoise(1))
	model_m.add(Conv1D(filters = 32, kernel_size = 2,))
	model_m.add(PReLU())
	model_m.add(BatchNormalization())
	model_m.add(MaxPool1D(2))
	model_m.add(Conv1D(filters = 32, kernel_size = 2, activation='relu'))
	model_m.add(BatchNormalization())
# 	model_m.add(Conv1D(filters = 32, kernel_size = 4, activation='relu'))
# 	model_m.add(BatchNormalization())
	model_m.add(Conv1D(filters = 32, kernel_size = 3, activation='relu'))
	model_m.add(BatchNormalization())	 
	model_m.add(MaxPool1D(2))
# 	model_m.add(Dropout(0.2))
	model_m.add(Flatten())
	
	model_r = Sequential()
	model_r.add(embedding_layer_r)
	model_r.add(GRU(cfg.input_classify_vocab_size, dropout=0.1, implementation = 2, 
						input_shape=(cfg.max_right_input_len, cfg.dim_right_embedding)))
	
	
	merged = Merge([model_l, model_m, model_r], mode='concat')
	model = Sequential()
	model.add(merged)
# 	model_l.add(DropconnectDense(units=512, rate=0.15, activation='relu'))
# 	model_l.add(DropconnectDense(units=1024, rate=0.25, activation='sigmoid'))
	
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.15))
# 	model.add(Dense(256, activation='sigmoid'))
# 	model.add(Dropout(0.25))
	
	model.add(Dense(3, activation='softmax'))
	
# 	model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
	model.compile(loss='categorical_crossentropy', optimizer = 'Adadelta', metrics=["accuracy"])
	return model

def gen_model_class_2():
	embedding_len = cfg.input_vocab_size / 2
	
	embedding_layer = Embedding(input_dim = cfg.input_vocab_size,  
                            output_dim = embedding_len,  
                            input_length=cfg.max_input_len,
                            embeddings_initializer = 'glorot_uniform',
#                             weights=[embedding_matrix],  
                            trainable=True)
	model = Sequential()
	model.add(embedding_layer)
# 	model1.add(GaussianNoise(1))
	model.add(Conv1D(filters = 32, kernel_size = 2, input_shape=(embedding_len, cfg.input_classify_vocab_size)))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Conv1D(filters = 32, kernel_size = 2, activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', kernel_regularizer=l2(0.01)))
	model.add(BatchNormalization())
	model.add(MaxPool1D(2))
#  	model.add(Dropout(0.1))
 	
# 	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu'))
# 	model.add(BatchNormalization())
# 	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu',  ))
# 	model.add(BatchNormalization())
# 	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu',  kernel_regularizer=l2(0.01)))
# 	model.add(BatchNormalization())	 
# 	model.add(MaxPool1D(2))
# 	model.add(Dropout(0.1))
	
	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu'))
	model.add(BatchNormalization())
# 	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu',  ))
# 	model.add(BatchNormalization())
	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu',  kernel_regularizer=l2(0.01)))
	model.add(BatchNormalization())	 
	model.add(MaxPool1D(2))
	model.add(Dropout(0.1))
	
	model.add(Flatten())
	model.add(Dense(128, activation='relu',))
	model.add(Dropout(0.1))
	model.add(Dense(256, activation='sigmoid', kernel_regularizer=l2(0.01), kernel_constraint=maxnorm(2.), bias_regularizer=l2(0.01)))
	model.add(Dropout(0.25))
	
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer = 'Adagrad', metrics=["accuracy"])
	return model
def gen_model_class_3():
	embedding_len = cfg.input_vocab_size / 2
	
	embedding_layer = Embedding(input_dim = cfg.input_vocab_size,  
                            output_dim = embedding_len,  
                            input_length=cfg.max_input_len,
                            embeddings_initializer = 'glorot_uniform',
#                             weights=[embedding_matrix],  
                            trainable=True)
	model = Sequential()
	model.add(embedding_layer)
# 	model1.add(GaussianNoise(1))
	model.add(Conv1D(filters = 32, kernel_size = 2, input_shape=(embedding_len, cfg.input_classify_vocab_size)))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Conv1D(filters = 32, kernel_size = 2, activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', kernel_regularizer=l2(0.01)))
	model.add(BatchNormalization())
	model.add(MaxPool1D(2))
#  	model.add(Dropout(0.1))
 	
# 	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu'))
# 	model.add(BatchNormalization())
# 	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu',  ))
# 	model.add(BatchNormalization())
# 	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu',  kernel_regularizer=l2(0.01)))
# 	model.add(BatchNormalization())	 
# 	model.add(MaxPool1D(2))
# 	model.add(Dropout(0.1))
	
	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu'))
	model.add(BatchNormalization())
# 	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu',  ))
# 	model.add(BatchNormalization())
	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu',  kernel_regularizer=l2(0.01)))
	model.add(BatchNormalization())	 
	model.add(MaxPool1D(2))
	model.add(Dropout(0.1))
	
	model.add(Flatten())
	
# 	model_extend = Sequential()
# 	model_extend.add(Input(shape=(5,)))
# 	model_extend.add(Dense(5, activation='relu',))
	model_extend = Sequential()
	model_extend.add(Dense(5, input_dim=5))

	merged = Merge([model, model_extend], mode='concat')
	
	modelc = Sequential()
	modelc.add(merged)
	modelc.add(Dense(128, activation='relu',))
	modelc.add(Dropout(0.1))
	modelc.add(Dense(64, activation='sigmoid', kernel_regularizer=l2(0.01), kernel_constraint=maxnorm(2.), bias_regularizer=l2(0.01)))
	modelc.add(Dropout(0.25))
	modelc.add(Dense(2, activation='softmax'))
	modelc.compile(loss='categorical_crossentropy', optimizer = 'Adagrad', metrics=["accuracy"])
	return modelc

def gen_model_class_4():
	embedding_len = cfg.input_vocab_size / 2
	
	embedding_layer = Embedding(input_dim = cfg.input_vocab_size,  
                            output_dim = embedding_len,  
                            input_length=cfg.max_input_len,
                            embeddings_initializer = 'glorot_uniform',
#                             weights=[embedding_matrix],  
                            trainable=True)
	model = Sequential()
	model.add(embedding_layer)
# 	model1.add(GaussianNoise(1))
	model.add(Conv1D(filters = 32, kernel_size = 2, input_shape=(embedding_len, cfg.input_classify_vocab_size)))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(MaxPool1D(2))
	model.add(Conv1D(filters = 32, kernel_size = 2, activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPool1D(2))
	model.add(Conv1D(filters = 32, kernel_size = 2, activation='relu', kernel_regularizer=l2(0.01)))
	model.add(BatchNormalization())
	model.add(MaxPool1D(2))
#  	model.add(Dropout(0.1))
 	
# 	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu'))
# 	model.add(BatchNormalization())
# 	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu',  ))
# 	model.add(BatchNormalization())
# 	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu',  kernel_regularizer=l2(0.01)))
# 	model.add(BatchNormalization())	 
# 	model.add(MaxPool1D(2))
# 	model.add(Dropout(0.1))
	
	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu'))
	model.add(BatchNormalization())
# 	model.add(MaxPool1D(2))
# 	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu',  ))
# 	model.add(BatchNormalization())
	model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu',  kernel_regularizer=l2(0.01)))
	model.add(BatchNormalization())	 
	model.add(MaxPool1D(2))
	model.add(Dropout(0.1))
	
	model.add(Flatten())
	model.add(Dense(128, activation='relu',))
	model.add(Dropout(0.1))
	model.add(Dense(256, activation='sigmoid', kernel_regularizer=l2(0.01), kernel_constraint=maxnorm(2.), bias_regularizer=l2(0.01)))
	model.add(Dropout(0.25))
	
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer = 'Adagrad', metrics=["accuracy"])
	return model
	
def gen_model_0():
	model = Sequential()
	model.add(Masking(mask_value=0, input_shape=(cfg.max_input_len, cfg.input_vocab_size)))
	model.add(LSTM(cfg.input_vocab_size, dropout=0.1, implementation = 2, input_shape=(cfg.max_input_len, cfg.input_vocab_size)))
	model.add(RepeatVector(cfg.max_output_len))
	model.add(LSTM(cfg.input_vocab_size, dropout=0., return_sequences=True, implementation = 2))
	model.add(TimeDistributed(Dense(cfg.output_vocab_size, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def gen_model_5():
	model = Sequential()
	model.add(Masking(mask_value=0, input_shape=(cfg.max_input_len, cfg.input_vocab_size)))
	model.add(Bidirectional(LSTM(cfg.input_vocab_size, dropout=0.1, implementation = 2, input_shape=(cfg.max_input_len, cfg.input_vocab_size)), merge_mode='sum'))
	model.add(RepeatVector(cfg.max_output_len))
	model.add(LSTM(cfg.input_vocab_size, dropout=0.1, implementation = 2, return_sequences=True))
	model.add(TimeDistributed(Dense(cfg.output_vocab_size, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# def experiment_tf():
# 	x_train, y_train, x_valid, y_valid = data_process.gen_data_from_npz("../data/train_filted.npz")
# 	
def loadModels(path, model):
	return model.load_weights(path)
# 
# 
# def classify(df):
# 	model = loadModel('../model/xgb_model')
# 	data = data_process.gen_features(df)
# 
# def experiment_test():
# 	
# 	x_train, y_train, x_valid, y_valid = data_process.gen_data_from_npz("../data/train.npz")
# 	
# 	# LSTM with Dropout for sequence classification in the IMDB dataset
# 
# 	# fix random seed for reproducibility
# 	np.random.seed(7)
# 	# load the dataset but only keep the top n words, zero the rest
# 	top_words = 5000
# 	(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# 	# truncate and pad input sequences
# 	max_review_length = 500
# 	X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# 	X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# 	# create the model
# 	embedding_vecor_length = 32
# 	model = Sequential()
# 	model.add(Embedding(input_dim=cfg.input_vocab_size,  
# 							  output_dim=cfg.input_hidden_dim,  
# 							  input_length=cfg.max_input_len))
# 	model.add(LSTM(cfg.input_hidden_dim, return_sequences=True))
# 	model.add(AttentionDecoder(hidden_dim=cfg.input_hidden_dim, output_dim=cfg.input_hidden_dim  
#                                              , output_length=cfg.max_output_len, state_input=False, return_sequences=True))  
# 	model.add(TimeDistributedDense(output_dim))  
# 	model.add(Activation('softmax'))
# 	
# 	model.add(Dropout(0.2))
# 	model.add(LSTM(100))
# 	model.add(Dropout(0.2))
# 	model.add(Dense(1, activation='softmax'))
# 	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	
# 	model = AttentionSeq2Seq(input_dim=5, input_length=7, hidden_dim=10, output_length=8, output_dim=20, depth=4)
# 	model.compile(loss='mse', optimizer='rmsprop')
# 	print(model.summary())
# 	
# 	model.fit(X_train, y_train, epochs=3, batch_size=64)
# 	# Final evaluation of the model
# 	scores = model.evaluate(X_test, y_test, verbose=0)
# 	print("Accuracy: %.2f%%" % (scores[1]*100))

def export_feature_data():

	df_train = pd.read_csv('../data/en_train_filted_all.csv')
	x_t, y_t = data_process.gen_features(df_train)
	
	print len(x_t)
	print len(y_t)
# 	print len(x_valid)
# 	print len(y_valid)


# 	
# 	
# 	coo_x_t = sparse.coo_matrix(x_t)
# 	coo_y_t = sparse.coo_matrix(y_t)
# 	coo_x_valid = sparse.coo_matrix(x_valid)
# 	coo_y_valid = sparse.coo_matrix(y_valid)
# 	sparse.mmwrite('coo_x_t.mtx', coo_x_t)
	np.savez("../data/train_filted.npz", x_t = x_t, y_t = y_t)

# def split_train_data():
# 	
# 	df = pd.read_csv('../data/en_train_filted_small.csv')
# 	df['id'] = df.apply(lambda x: x['sentence_id'] + "_" + x['token_id'], axis=1)
# 	all_index = df.index.tolist()
# 	x_train, x_valid, y_train, y_valid = train_test_split(df, y_t,
#                                                        test_size=0.1, random_state=2017)
# 	
# 	vaild_index = random.sample(range(0, len(df)), int(0.1 * len(df)));
# 	train_index = list(set(all_index) ^ set(vaild_index))
# 	
# 	df_train = df.iloc[train_index]
# 	df_vaild = df.iloc[vaild_index]
def experiment_attention4():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz")
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_2(depth=4, dropout=0.3)
	print(model.summary())
	
	train_sparse(model, "attention4_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=256, nb_epoch = 500)
	
	
def experiment_attention3():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz")
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_2(depth=3, dropout=0.25)
	print(model.summary())
	
	train_sparse(model, "attention3_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=256, nb_epoch = 300)
	
def experiment_attention1():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz")
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_2(depth=1)
	print(model.summary())
	
	train_sparse(model, "attention1_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=64, nb_epoch = 50)
	
def experiment_simple1():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz")
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_0()
	print(model.summary())
	
	train_sparse(model, "simple1_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=256, nb_epoch = 50)
	
def experiment_simple1_bi():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz")
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_5()
	print(model.summary())
	
	train_sparse(model, "simple1_bi_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=256, nb_epoch = 50)
	
def experiment_simple2():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz")
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_3(depth=2, peek=True, teacher_force=False)
	print(model.summary())
	
	train_sparse(model, "simple2_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=256, nb_epoch = 150)
	
	
def experiment_simple3():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz")
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_3(depth=3, dropout=0.2, peek=True, teacher_force=False)
	print(model.summary())
	
	train_sparse(model, "simple3_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=256, nb_epoch = 200)
	
	
def experiment_teaching1():
	
	data = np.load("../data/train_cls1.npz")
	x_t_c = data['x_t_c']
	y_t = data['y_t']
	
	x_train, x_valid, y_train, y_valid = train_test_split(x_t_c, y_t, test_size=10000, random_state=0)
	model = gen_model_teaching_0()
	print(model.summary())
	
# 	model = AttentionSeq2Seq(input_dim=cfg.max_input_len, hidden_dim=cfg.input_hidden_dim, output_length=cfg.max_output_len, output_dim=cfg.output_vocab_size, depth=2)
# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
# 	print(model.summary())
	train_teaching_onehot(model, "teach_l1_l1", x_train, y_train, x_valid, y_valid, 
					batch_size=256, nb_epoch = 100)
	
# def experiment_teaching2():
# 	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz")
# 	x_t = sparse.csr_matrix(x_train)
# # 	x_v = sparse.csr_matrix(x_valid)
# 	x_v = x_valid
# 	
# 	model = gen_model_3(depth=2, dropout=0.3, peek=True, teacher_force=True)
# 	print(model.summary())
# 	
# 	train_sparse_teaching(model, "teach2_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
# 				batch_size=256, nb_epoch = 150)
	
	
def experiment_classify():
	x_train, y_train, x_valid, y_valid, _, _ = data_process.gen_data_from_npz("../data/en_train_classify.npz", True)	
# 	np.savez("../data/en_valid_classify.npz", x_t_cls = x_valid, y_t_cls=y_valid)
	model = gen_model_class_0()
	print(model.summary())
	
	train_classify(model, "class_cnn", x_train, y_train, x_valid, y_valid, 
				batch_size=256, nb_epoch = 200)
	
def experiment_classify_char():
	y_t = data_process.load_numpy_data("../data/train_y_2class.npy")
	
	data = np.load("../data/en_train.npz")
	x_t_c = data['x_t_c']

	x_train, x_valid, y_train, y_valid = train_test_split(x_t_c, y_t, test_size=200000, random_state=0)
	model = gen_model_class_4()
	print(model.summary())
	train_classify(model, "class_cnn_c4", x_train, y_train, x_valid, y_valid, 
			batch_size=256, nb_epoch = 100)
	
def experiment_classify_char_and_extend():

	y_t = data_process.load_numpy_data("../data/train_y_2class.npy")
	
	data = np.load("../data/en_train.npz")
	x_t_c = data['x_t_c']
	
	x_t_e = data_process.load_numpy_data("../data/extend_features.npy")
	
	
	x_train, x_valid, y_train, y_valid = train_test_split(np.hstack([x_t_c, x_t_e]), y_t, test_size=10, random_state=0)
	model = gen_model_class_3()
	print(model.summary())
	train_classify_extend(model, "class_cnn_c3", x_train, y_train, x_valid, y_valid, 
			batch_size=256, nb_epoch = 100)
	
def experiment_classify_frag():
	data = np.load("../data/en_train_classify.npz")
	y_t = data['y_t']
	data = np.load("../data/en_train_frag_char.npz")
	x_char_l=data['x_char_l']
	x_char_m=data['x_char_m']
	x_char_r=data['x_char_r']
	
	x_train, x_valid, y_train, y_valid = train_test_split(np.hstack([x_char_l, x_char_m, x_char_r]), y_t, test_size=10000, random_state=0)
	model = gen_model_class_1()
	print(model.summary())
	
	train_classify_frag(model, "class_rcnn", x_train, y_train, x_valid, y_valid, 
			batch_size=256, nb_epoch = 200)
	
def run_experiments():
	experiment_simple1()
	experiment_simple2()
	experiment_simple3()
	experiment_attention3()
	experiment_attention4()
	experiment_teaching1()
# 	experiment_teaching2()
	
def evalute_acc(ret_file, err_file):
	df_ret = pd.read_csv(ret_file)
	df_c = df_ret[df_ret['after_truth']==df_ret['after']]
	df_err = df_ret[df_ret['after_truth']!=df_ret['after']]
	df_err.to_csv(err_file)
	print 'The corrected num is:%d, real acc:%f'%(len(df_c), len(df_c)/float(len(df_ret)))

def split_by_classifier(classifier, x_t_cls, x_t, df_test, y_t_cls=None, y_t=None):
	predict_y = classifier.predict(x_t_cls, batch_size = 256, verbose=0)
	p_y = np.argmax(predict_y, axis=1)
	x_list = []
	cls_df_list = []
	index0 = np.where(p_y==0)[0].tolist()
	index1 = np.where(p_y==1)[0].tolist()
# 	index2 = np.where(p_y==2)[0].tolist()
	
	x_list.append(x_t[index0])
	x_list.append(x_t[index1])
# 	x_list.append(x_t[index2])
	
	cls_df_list.append(df_test.iloc[index0])
	cls_df_list.append(df_test.iloc[index1])
# 	cls_df_list.append(df_test.iloc[index2])
	
	cls_df_list[0]['p_new_class'] = 1
	cls_df_list[1]['p_new_class'] = 2
# 	cls_df_list[2]['p_new_class'] = 3
	
	cls_y_list = []
	if y_t_cls is not None:
		cls_y_list.append(y_t_cls[index0])
		cls_y_list.append(y_t_cls[index1])
# 		cls_y_list.append(y_t_cls[index2])
		
	y_list = []
	if y_t is not None:
		y_list.append(y_t[index0])
		y_list.append(y_t[index1])
# 		y_list.append(y_t[index2])
		
	
	return x_list, cls_df_list, cls_y_list, y_list


def run_normalize():
	df_test = pd.read_csv('../data/en_test.csv')
	test_feat = np.load('../data/test_classify.npz')
	x_t_cls = test_feat['x_t_cls']
	x_test = test_feat['x_t']
	
	
	
	mod_classifier = gen_model_class_2()
	mod_classifier.load_weights("../model/class_cnn_c2_weights.98-0.0005-1.0000-0.0004-1.0000.hdf5")

	
	cls_x_list, cls_df_list, _, _ = split_by_classifier(mod_classifier, x_t_cls, x_test, df_test)
	
	
# 	nomalizer1 = loadModel("../model/nomalizer1.mod")
# 	nomalizer2 = loadModel("../model/nomalizer2.mod")
# 	nomalizer3 = loadModel("../model/nomalizer3.mod")
# 	cls_list[1] = normalize(nomalizer1, cls_list[1])
# 	cls_list[2] = normalize(nomalizer1, cls_list[2])
# 	cls_list[3] = normalize(nomalizer1, cls_list[3])
	
def run_evalute_and_split():
	df_test = pd.read_csv('../data/train_filted_classify.csv')
	df_test['p_new_class'] = -1
# 	test_feat = np.load('../data/en_train_classify.npz')
	y_t_cls = data_process.load_numpy_data('../data/train_y_2class.npy')
	
	test_feat = np.load("../data/en_train.npz")
	
	x_t_cls = test_feat['x_t_c']
	 
	
	test_feat = np.load('../data/en_train.npz')
	x_t = test_feat['x_t']
	y_t = test_feat['y_t']
	
	x_t_e = np.load('../data/extend_features.npy')
	
# 	x_t_cls = [x_t_cls, x_t_e]
	
	mod_classifier = gen_model_class_2()
	print(mod_classifier.summary())
	mod_classifier.load_weights("../model/class_cnn_c2_weights.98-0.0005-1.0000-0.0004-1.0000.hdf5")
	x_list, cls_df_list, cls_y_list, y_list = split_by_classifier(mod_classifier, x_t_cls, np.hstack([x_t, x_t_cls]), df_test, y_t_cls, y_t)
	
	
	correct_num1 = len(cls_df_list[0][cls_df_list[0]['new_class1']==1])
	correct_num2 = len(cls_df_list[1][cls_df_list[1]['new_class1']==2])
	classify_acc0 = correct_num1 / float(len(cls_df_list[0]))
	classify_acc1 = correct_num2 / float(len(cls_df_list[1]))
# 	classify_acc2 = len(cls_df_list[2][cls_df_list[2]['new_class']==cls_df_list[2]['p_new_class']]) / float(len(cls_df_list[2]))
	
	df_out = pd.concat(cls_df_list)
	
	classify_acc = (correct_num1 + correct_num2) / float(len(df_out))
	df_out_err = df_out[df_out['new_class1']!=df_out['p_new_class']]
	
	print "Total item num:%d, c1 num:%d, c2 num:%d, err num:%d"%(len(df_out), len(cls_df_list[0]), len(cls_df_list[1]), len(df_out_err))
	print "Total classify acc:%f, c1 classify acc:%f, c2 classify acc:%f"%(classify_acc, classify_acc0, classify_acc1)
	df_out.to_csv('../data/classified.csv', index=False)
	df_out_err.to_csv('../data/classified_err.csv', index=False)
	
	cls_df_list[0].to_csv('../data/train_cls1.csv', index=False)
	cls_df_list[1].to_csv('../data/train_cls2.csv', index=False)
	np.savez("../data/train_cls1.npz", x_t = x_list[0][:, :cfg.max_input_len], x_t_c = x_list[0][:, cfg.max_input_len:], y_t=y_list[0])
	np.savez("../data/train_cls2.npz", x_t = x_list[1][:, :cfg.max_input_len], x_t_c = x_list[1][:, cfg.max_input_len:], y_t=y_list[1])
	
def display_model():
	model = gen_model_class_0()
	print(model.summary())
	model.load_weights("../model/class_cnn1_1_weights.24-0.0027-0.9992-0.0266-0.9958.hdf5")
	layer1 = model.get_layer('conv_1')
	w1 = layer1.get_weights()
	print w1
	
if __name__ == "__main__":
# 	data_process.gen_alpha_table()
# 	run_evalute()
# 	data_process.add_class_info('../data/en_test.csv', "../data/en_test_class.csv")
# 	df = pd.read_csv('../data/en_train_filted_class.csv')
# 	data_process.add_class_info('../data/en_train_filted_all.csv', '../data/en_train_filted_class.csv')
# 	data_process.gen_train_feature(df)
# 	data_process.gen_extend_features(df)
# 	print len(df)
#  	df['len'] = df['before'].apply(lambda x:len(str(x)))
#  	df_c1 = df[df['new_class'] == 1]
#  	c1_cnt = df_c1['len'].value_counts().sort_values()
#  	c1_cnt.to_csv('../data/c1_cnt.csv', index=True)
#  	df_c2 = df[df['new_class'] == 2]
#  	c2_cnt = df_c2['len'].value_counts().sort_values()
#  	c2_cnt.to_csv('../data/c2_cnt.csv', index=True)
#  	df_c3 = df[df['new_class'] == 3]
#  	c3_cnt = df_c3['len'].value_counts().sort_values()
#  	c3_cnt.to_csv('../data/c3_cnt.csv', index=True)
 	experiment_teaching1()
#  	del df['len']
#  	print 'saved cnt info.'
#  	data_process.gen_one2one_dict(df)
	
# 	evalute_acc('../data/test_ret.csv', '../data/test_ret_err.csv')
# 	export_feature_data()
# 	run_evalute()
#  	run_evalute_and_split()
# 	experiment_classify_char_and_extend()
# 	t = fst.Transducer()
# 	t.add_arc(0, 1, 'a', 'A')
# 	t.add_arc(0, 1, 'b', 'B')
# 	t.add_arc(1, 2, 'c', 'C')
# 	
# 	t[2].final = True
# 	
# 	print t.shortest_path()
# 	t.write("data/fst.txt")
	
# 	se = pd.Series(['<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>'])
# 	ps = pd.DataFrame(columns=['word'])
# 	ps['word'] = se
# 	print ps
# 	df = pd.read_csv('../data/en_train_filted_class.csv') 
# 	data_process.display_sentence(218309, df)
# 	
# 	data_process.display_sentence(110195, df)

