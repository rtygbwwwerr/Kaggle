import pandas as pd
import data_process
import numpy as np
import fst
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Input, Dense, Masking
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.engine.training import Model
from keras.preprocessing import sequence
from keras.callbacks import LearningRateScheduler,EarlyStopping,TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from scipy.sparse import csr_matrix
import tensorflow as tf
import keras
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
			
def sparse_generator_teaching(X, y, batch_size=128, shuffle=True):
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
		yield [X_batch, y_batch], y_batch
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

def reshape_data_teaching(X, y):
	X_out = np.zeros((X.shape[0], X.shape[1], cfg.input_vocab_size))
	for i in range(X.shape[0]):
		for t in range(X.shape[1]):
			X_out[i,t,X[i,t]] = 1.0
	
	y_out = np.zeros((y.shape[0], y.shape[1], cfg.output_vocab_size))
	for i in range(y.shape[0]):
		for t in range(y.shape[1]):
			y_out[i,t,y[i,t]] = 1.0
				
	return [X_out, y_out], y_out

def reshape_data_two(X, y):
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

def train_sparse_teaching(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, df_valid, batch_size=128, nb_epoch = 100):
	
	# reshape X to be [samples, time steps, features]
# 	X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
# 	Y_valid = np.reshape(Y_valid, (Y_valid.shape[0], Y_valid.shape[1], 1))
	X_valid, Y_valid = reshape_data_teaching(X_valid, Y_valid)
# 	print X_train.getnnz()
# 	print X_train.shape[0]
# 	print X_train.shape[1]

	board = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s%s_%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%("LSTM2LSTM2in","1","1")
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
#  	samples_per_epoch = batch_size
	model.fit_generator(generator=sparse_generator_teaching(X_train, Y_train, batch_size, False), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
			    	    validation_data=(X_valid, Y_valid),
			    		callbacks=[board, checkpointer])
	print 'Training time', time.time() - start_time
	# evaluate network
# 	decode_sequence(X_valid, Y_valid, extend_args)
	
	score = model.evaluate(X_valid, Y_valid, batch_size)
	p_y = model.predict(X_valid, batch_size, verbose=0)
	np.savez(ret_file_head + '.npz', p_y = p_y)
	out_list = get_predict_list(p_y)
	out_list = format_out_list(out_list)
	output_to_csv(out_list, df_valid, ret_file_head + '.csv')
	print out_list
# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)

def train_sparse_two(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, df_valid, batch_size=128, nb_epoch = 100):
	
	# reshape X to be [samples, time steps, features]
# 	X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
# 	Y_valid = np.reshape(Y_valid, (Y_valid.shape[0], Y_valid.shape[1], 1))
	X_valid, X_d_vaild, Y_valid = reshape_data_two(X_valid, Y_valid)
# 	print X_train.getnnz()
# 	print X_train.shape[0]
# 	print X_train.shape[1]

	board = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s%s_%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%("LSTM2LSTM2in","1","1")
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
#  	samples_per_epoch = batch_size
	model.fit_generator(generator=sparse_generator_two(X_train, Y_train, batch_size, False), 
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
	np.savez(ret_file_head + '.npz', p_y = p_y)
	out_list = get_predict_list(p_y)
	out_list = format_out_list(out_list)
	output_to_csv(out_list, df_valid, ret_file_head + '.csv')
	print out_list
# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)

def decode_sequence(input_sequence, out_sequence, extend_args):
	
	latent_dim = cfg.input_vocab_size
	num_encoder_tokens = cfg.input_vocab_size
	num_decoder_tokens = cfg.output_vocab_size
	
	encoder_model = Model(extend_args['encoder_inputs'], extend_args['encoder_states'])
	decoder_state_input_h = Input(shape=(latent_dim,))
	decoder_state_input_c = Input(shape=(latent_dim,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = extend_args['decoder_lstm'](
	    extend_args['decoder_inputs'], initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = extend_args['decoder_dense'](decoder_outputs)
	decoder_model = Model(
	    [extend_args['decoder_inputs']] + decoder_states_inputs,
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
	check_file = "../checkpoints/%s%s_%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%("LSTM2LSTM","1","1")
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
# 	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)) * batch_size)
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
	model.fit_generator(generator=sparse_generator(X_train, Y_train, batch_size, False), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
			    	    validation_data=(X_valid, Y_valid),
			    	    callbacks=[board, checkpointer]
			    		)
	print 'Training time', time.time() - start_time
	# evaluate network
	score = model.evaluate(X_valid, Y_valid, batch_size)
	p_y = model.predict(X_valid, batch_size, verbose=0)
	np.savez(ret_file_head + ".npz", p_y = p_y)
	out_list = get_predict_list(p_y)
	out_list = format_out_list(out_list)
	output_to_csv(out_list, df_valid, ret_file_head + ".csv")
	print out_list

# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)
	
# def acc_true(y_true, y_pred):
# 	cnt = 0.0
# 	tensors1 = y_true.unpack(value = y_true, axis=0)
# 	tensors2 = tf.unpack(value = y_pred, axis=0)
# 	for yt, yp in y_true, y_pred:
# 		if (yt==yp).all():
# 			cnt = cnt + 1.0
	
	
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

def gen_model_4():
	model = Sequential()

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

def gen_model_1():
	# Define an input sequence and process it.
	encoder_inputs = Input(shape=(None, cfg.input_vocab_size))
	encoder = LSTM(cfg.input_vocab_size, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]
	
	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(None, cfg.output_vocab_size))
	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the
	# return states in the training model, but we will use them in inference.
	decoder_lstm = LSTM(cfg.input_vocab_size, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(cfg.output_vocab_size, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	
	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 	extend_args = {'encoder_inputs':encoder_inputs, "encoder_states":encoder_states, "decoder_inputs":decoder_inputs,
# 				   'decoder_lstm':decoder_lstm, 'decoder_dense':decoder_dense }
	return model

def gen_model_0():
	model = Sequential()
	model.add(Masking(mask_value=0, input_shape=(cfg.max_input_len, cfg.input_vocab_size)))
	model.add(LSTM(384, input_shape=(cfg.max_input_len, cfg.input_vocab_size)))
	model.add(RepeatVector(cfg.max_output_len))
	model.add(LSTM(384, return_sequences=True))
	model.add(TimeDistributed(Dense(cfg.output_vocab_size, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def experiment():
# 	df_train = pd.read_csv("../data/en_train.csv", nrows=1000)
# 	df_test = pd.read_csv("../data/en_test.csv", nrows=1000)
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz", range=None)
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
# 	model = gen_model_1()
	model, extend_args = gen_model_1()
	print(model.summary())
	
# 	model = AttentionSeq2Seq(input_dim=cfg.max_input_len, hidden_dim=cfg.input_hidden_dim, output_length=cfg.max_output_len, output_dim=cfg.output_vocab_size, depth=2)
# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
# 	print(model.summary())
	train_sparse_two(model, "../data/test_ret.npz", x_t, y_train, x_v, y_valid, df_valid, batch_size=256)
# 	model.fit(x_train, y_train, epochs=3, batch_size=64)

# def experiment_tf():
# 	x_train, y_train, x_valid, y_valid = data_process.gen_data_from_npz("../data/train_filted.npz", range=None)
# 	
def loadModel(path):
	return None
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
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz", range=None)
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_2(depth=4, dropout=0.3)
	print(model.summary())
	
	train_sparse(model, "../data/attention4_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=256, nb_epoch = 500)
	
	
def experiment_attention3():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz", range=None)
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_2(depth=3, dropout=0.25)
	print(model.summary())
	
	train_sparse(model, "../data/attention3_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=256, nb_epoch = 400)

def experiment_simple1():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz", range=None)
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_0()
	print(model.summary())
	
	train_sparse(model, "../data/simple1_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=256, nb_epoch = 3)
	
	
def experiment_simple2():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz", range=None)
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_3(depth=2, peek=True, teacher_force=False)
	print(model.summary())
	
	train_sparse(model, "../data/simple2_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=256, nb_epoch = 150)
	
	
def experiment_simple3():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz", range=None)
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_3(depth=3, dropout=0.2, peek=True, teacher_force=False)
	print(model.summary())
	
	train_sparse(model, "../data/simple3_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=256, nb_epoch = 200)
	
	
def experiment_teaching1():
# 	df_train = pd.read_csv("../data/en_train.csv", nrows=1000)
# 	df_test = pd.read_csv("../data/en_test.csv", nrows=1000)
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz", range=None)
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
# 	model = gen_model_1()
	model = gen_model_1()
	print(model.summary())
	
# 	model = AttentionSeq2Seq(input_dim=cfg.max_input_len, hidden_dim=cfg.input_hidden_dim, output_length=cfg.max_output_len, output_dim=cfg.output_vocab_size, depth=2)
# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
# 	print(model.summary())
	train_sparse_two(model, "../data/teach_test_ret.npz", x_t, y_train, x_v, y_valid, df_valid, 
					batch_size=256, nb_epoch = 100)
	
def experiment_teaching2():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz", range=None)
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_3(depth=2, dropout=0.3, peek=True, teacher_force=True)
	print(model.summary())
	
	train_sparse_teaching(model, "../data/teach2_test_ret", x_t, y_train, x_v, y_valid, df_valid, 
				batch_size=256, nb_epoch = 150)
	
	
	
	
def run_experiments():
	experiment_simple1()
	experiment_simple2()
	experiment_simple3()
	experiment_attention3()
	experiment_attention4()
	experiment_teaching1()
	experiment_teaching2()
	
def evalute_acc(ret_file, err_file):
	df_ret = pd.read_csv(ret_file)
	df_c = df_ret[df_ret['after_truth']==df_ret['after']]
	df_err = df_ret[df_ret['after_truth']!=df_ret['after']]
	df_err.to_csv(err_file)
	print 'The corrected num is:%d, real acc:%f'%(len(df_c), len(df_c)/float(len(df_ret)))
	
if __name__ == "__main__":
	experiment_attention4()
# 	evalute_acc('../data/test_ret.csv', '../data/test_ret_err.csv')
# 	export_feature_data()
# 	experiment()
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
	

