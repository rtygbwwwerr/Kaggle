import codecs
import csv
import re
import pandas as pd
import data_process
from base.data_seq import NumpySeqData
import numpy as np
#import fst
from keras.datasets import imdb
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense, Masking, Merge, Permute
from keras.layers import LSTM, GRU
from keras.layers.core import Reshape,Activation
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
#from bokeh.server.protocol.messages import index
from bokeh.util.session_id import random
from keras.layers.wrappers import Bidirectional
import heapq
from operator import itemgetter
from beam_search import beam_search
from keras.layers import merge
import model_maker
from math import ceil, floor
#from vis.optimizer import Optimizer
from tensorflow.python import debug as tf_debug
from sklearn.cross_validation import train_test_split
import sys
from post_processor import RuleBasedNormalizer

reload(sys)
sys.setdefaultencoding('utf8')
from __builtin__ import str
cfg.init()
rule_norm_obj = RuleBasedNormalizer()

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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
			
def generator_onehot(X, y, batch_size=128, shuffle=True):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
		X_batch = np.zeros((batch_size, X.shape[1], cfg.vocab_size))
		y_batch = np.zeros((batch_size, y.shape[1], cfg.vocab_size))
		# reshape X to be [samples, time steps, features]
		for i, j in enumerate(batch_index):
# 			X_batch[i,:,:] = np.transpose(X[j].toarray())
			for t in range(X.shape[1]):
				X_batch[i,t,X[j, t]] = 1.0
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

def reshape_data_onehot(X, y):
	X_out = np.zeros((X.shape[0], X.shape[1], cfg.vocab_size))
	for i in range(X.shape[0]):
		for t in range(X.shape[1]):
			X_out[i,t,X[i,t]] = 1.0
	
	y_out = np.zeros((y.shape[0], y.shape[1], cfg.vocab_size))
	for i in range(y.shape[0]):
		for t in range(y.shape[1]):
			y_out[i,t,y[i,t]] = 1.0
					
	return X_out, y_out

def train_teaching(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, batch_size=128, nb_epoch = 100):
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
# 	decode_sequence_teach(X_valid, Y_valid, extend_args)
	
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

def train_teaching_onehot(model, log_dir, ret_file_head, X_train, Y_train, X_valid, Y_valid, initial_epoch, batch_size=128, nb_epoch = 100):

	# reshape X to be [samples, time steps, features]
# 	X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
# 	Y_valid = np.reshape(Y_valid, (Y_valid.shape[0], Y_valid.shape[1], 1))
	X_valid, X_d_vaild, Y_valid = reshape_data_teaching_onehot(X_valid, Y_valid)
# 	print X_train.getnnz()
# 	print X_train.shape[0]
# 	print X_train.shape[1]

	board = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head)
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=False)
	# start training
	start_time = time.time()
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
#  	samples_per_epoch = batch_size
	model.fit_generator(generator=generator_teaching_onehot(X_train, Y_train, batch_size, False), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
			    	    validation_data=([X_valid, X_d_vaild], Y_valid),
			    		callbacks=[board, checkpointer], 
			    		initial_epoch=initial_epoch)
	print 'Training time', time.time() - start_time
	# evaluate network
# 	decode_sequence_teach(X_valid, Y_valid, extend_args)
	
	score = model.evaluate([X_valid, X_d_vaild], Y_valid, batch_size)
	p_y = model.predict([X_valid, X_d_vaild], batch_size, verbose=0)
	out_list = get_predict_list(p_y)
	out_list = format_out_list(out_list)
# 	output_to_csv(out_list, df_valid, path + ret_file_head + '.csv')
	print out_list
# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)
def reshape_data_tf_copyNet(X, y):
	X_decode_batch = np.where(y==cfg.end_flg_index, cfg.pad_flg_index, y)
	y_batch = np.hstack([y[:,1:],np.ones((y.shape[0],1))*cfg.pad_flg_index])
 	y_copy_batch = np.zeros((y_batch.shape[0], y_batch.shape[1], cfg.vocab_size + cfg.max_input_len))
	indices = np.where(X==cfg.split_input_index)
	list_indices = []
	for i in range(0, indices[1].shape[0], 2):
		temp = list(indices[1])
		temp[0] = temp[0] + 1
# 		temp = map(lambda x:int(x), temp)
		list_indices.append(temp)

	for i in range(len(list_indices)):
		in_chars = list(X[i, list_indices[i][0]:list_indices[i][1]])
		out_chars = list(y_batch[i])
		for j in range(len(out_chars)):
			c = int(out_chars[j])
			y_copy_batch[i,j,c] = 1.0
# 			if c == cfg.end_flg_index or c == cfg.pad_flg_index:
# 				break
			for k in range(len(in_chars)):
				if c == int(in_chars[k]):
					index = cfg.vocab_size + list_indices[i][0] + k
					y_copy_batch[i,j,index] = 1.0
					y_copy_batch[i,j,c] = 0.0
					break
			
		
		
	return  X_decode_batch, y_copy_batch

def reshape_data_tf_teach_onehot(X, y):
	X_decode_batch = np.where(y==cfg.end_flg_index, cfg.pad_flg_index, y)
	y_batch = np.hstack([y[:,1:],np.ones((y.shape[0],1))*cfg.pad_flg_index])
	X_out = np.zeros((X.shape[0], X.shape[1], cfg.input_vocab_size))
	for i in range(X.shape[0]):
		for t in range(X.shape[1]):
			X_out[i,t,int(X[i,t])] = 1.0
	
	y_out = np.zeros((y.shape[0], y.shape[1], cfg.output_vocab_size))
	X_d_out = np.zeros((y.shape[0], y.shape[1], cfg.output_vocab_size))
	for i in range(y.shape[0]):
		for t in range(y.shape[1]):
			X_d_out[i,t,int(X_decode_batch[i,t])] = 1.0
			y_out[i,t,int(y_batch[i, t])] = 1.0
			
	return X_out, X_d_out, y_out

def reshape_data_tf_teach_onehot_y(X, y):
	X_decode_batch = np.where(y==cfg.end_flg_index, cfg.pad_flg_index, y)
	y_batch = np.hstack([y[:,1:],np.ones((y.shape[0],1))*cfg.pad_flg_index])
	X_out = X

	
	y_out = np.zeros((y.shape[0], y.shape[1], cfg.vocab_size))
	for i in range(y.shape[0]):
		for t in range(y.shape[1]):
			y_out[i,t,int(y_batch[i, t])] = 1.0
			
	return X_out, X_decode_batch, y_out
			
def reshape_data_tf_teach(X, y):
	X_decode_batch = np.where(y==cfg.end_flg_index, cfg.pad_flg_index, y)
	y_batch = np.hstack([y[:,1:],np.ones((y.shape[0],1))*cfg.pad_flg_index])
	X_out = X
	return X_out, X_decode_batch, y_batch

def get_batch_data_tf_teaching(X, y, batchid, batch_size, reshape_fn=reshape_data_tf_teach_onehot, shuffle=False):
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
		
	batch_index = sample_index[batch_size*batchid:batch_size*(batchid+1)]
	X_batch = X[batch_index,:]
	y_batch = y[batch_index,:]
	print "round{0},batch size:{1}".format(batchid, X_batch.shape[0])

	# reshape X to be [samples, time steps, features]
	X_out, X_d_out, y_out = reshape_fn(X_batch, y_batch)
	return X_out, X_d_out, y_out

def get_batch_data(X, y, batchid, batch_size, reshape_fn, shuffle=True):
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
		
	batch_index = sample_index[batch_size*batchid:batch_size*(batchid+1)]
	X_batch = X[batch_index,:]
	y_batch = y[batch_index,:]
	print "round{0},batch size:{1}".format(batchid, X_batch.shape[0])
# 	X_decode_batch = y[batch_index,:]
# 	#delete all end flag in decode inputs
# 	X_decode_batch = np.where(X_decode_batch==cfg.end_flg_index, cfg.pad_flg_index, X_decode_batch)
# 	
# 	#delete all start flag in target sequences
# 	y_batch = np.hstack([y_batch[:,1:],np.ones((batch_index,1))*cfg.pad_flg_index])
	# reshape X to be [samples, time steps, features]
	X_decode_batch, y_batch = reshape_data_tf_copyNet(X_batch, y_batch)
# 	encoder_inputs_lengths = np.sum(X_batch!=cfg.pad_flg_index, axis=1)
# 	decoder_inputs_lengths = np.sum(X_decode_batch!=cfg.pad_flg_index, axis=1)
	return X_batch, X_decode_batch, y_batch
			
def train_tf_teaching_copynet(model_fn, log_dir, ret_file_head, 
			X_train, Y_train, X_valid, Y_valid, 
			initial_epoch, batch_size=128, nb_epoch = 100):
# 	print X_train[0]
# 	print Y_train[0]
	
	# placeholder for inputs
	p_encoder_inputs = tf.placeholder(tf.int32, shape=(cfg.batch_size, None))
	p_decoder_inputs = tf.placeholder(tf.int32, shape=(cfg.batch_size, None))
	p_decoder_outputs = tf.placeholder(tf.int32, shape=(cfg.batch_size, cfg.max_output_len,None))
	
	# placeholder for sequence lengths
	p_encoder_inputs_lengths = tf.placeholder(tf.int32, shape=(cfg.batch_size,))
	p_decoder_inputs_lengths = tf.placeholder(tf.int32, shape=(cfg.batch_size,))
	args_input = {}
	args_input['encoder_inputs'] = p_encoder_inputs
	args_input['decoder_inputs'] = p_decoder_inputs
	args_input['decoder_outputs'] = p_decoder_outputs
	args_input['encoder_inputs_lengths'] = p_encoder_inputs_lengths
	args_input['decoder_inputs_lengths'] = p_decoder_inputs_lengths
	
	max_batchid = int(np.ceil(X_train.shape[0] / batch_size))
	
	with tf.variable_scope('root'):
		train_loss, train_op = model_fn(args_input, mode='train')
	
# 	with tf.variable_scope('root', reuse=True):
# 		eval_loss = model_fn(args_input, mode='eval')
	
# 	with tf.variable_scope('root', reuse=True):
# 		translations = model_fn(mode='infer')
	
	# session configure for GPU
	sess_config = tf.ConfigProto()
	# sess_config.gpu_options.allow_growth = True
	# sess_config.per_process_gpu_memory_fraction = 0.5
	with tf.Session(config=sess_config) as sess:
		saver = tf.train.Saver()
# 		sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# 		sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
		sess.run(tf.global_variables_initializer())
		for batchid in range(max_batchid):
			# training
# 			train_batch = train_data.next_batch()
			encoder_inputs, decoder_inputs, decoder_outputs = get_batch_data(X_train, Y_train, batchid, 
								batch_size, False)
			feed_dict = {
			    p_encoder_inputs: encoder_inputs,
			    p_decoder_inputs: decoder_inputs,
			    p_decoder_outputs:decoder_outputs,
				p_encoder_inputs_lengths: np.ones((batch_size,)) * cfg.max_input_len,
				p_decoder_inputs_lengths: np.ones((batch_size,)) * cfg.max_output_len
# 			    p_encoder_inputs_lengths: np.sum(encoder_inputs!=cfg.pad_flg_index, axis=1),
# 			    p_decoder_inputs_lengths: np.sum(decoder_inputs!=cfg.pad_flg_index, axis=1)
			} 
			loss_train, _ = sess.run([train_loss, train_op], feed_dict=feed_dict)
			print 'batchid = %d, train loss = %f' % (batchid, loss_train)
			
			# save checkpoints
			if (batchid + 1) % cfg.save_freq == 0:
				saver.save(sess, '../checkpoints/', global_step=batchid)
				print 'model saved'
			
# 			# eval
# 			if (batchid + 1) % cfg.eval_freq == 0:
# 				encoder_inputs = X_valid
# 				decoder_inputs, decoder_outputs = reshape_data_tf(Y_valid)
# 				feed_dict = {
# 				    p_encoder_inputs: encoder_inputs,
# 				    p_decoder_inputs: decoder_inputs,
# 				   	p_decoder_outputs: decoder_outputs,
# 				    p_encoder_inputs_lengths: np.sum(encoder_inputs!=cfg.pad_flg_index, axis=1),
# 				    p_decoder_inputs_lengths: np.sum(decoder_inputs!=cfg.pad_flg_index, axis=1)
# 				}
# 				loss_eval = sess.run(eval_loss, feed_dict=feed_dict)
# 				print '\t\tbatchid = %d, eval loss = %f' % (batchid, loss_eval)
			
# 			# inference
# 			if (batchid + 1) % cfg.infer_freq == 0:
# 				encoder_inputs, decoder_inputs, decoder_outputs = get_batch_data(X_valid, Y_valid, batchid, 
# 								batch_size, False)
# 				feed_dict = {
# 				    encoder_inputs: encoder_inputs,
# 				    decoder_inputs: decoder_inputs,
# 				    decoder_outputs: decoder_outputs,
# 				    encoder_inputs_lengths: np.sum(encoder_inputs!=cfg.pad_flg_index, axis=1),
# 				    decoder_inputs_lengths: np.sum(decoder_inputs!=cfg.pad_flg_index, axis=1)
# 				}
# 				word_ids = sess.run(translations, feed_dict=feed_dict)
# 				print 'batchid = %d' % (batchid)
# 				index = random.randint(0, 39)
# 				print 'input:', vocab.word_ids_to_sentence(
# 				        train_batch.encoder_inputs[index]), '\n'
# 				print 'output:', vocab.word_ids_to_sentence(
# 				        train_batch.decoder_inputs[index]), '\n'
# 				print 'predict:', vocab.word_ids_to_sentence(
# 				        word_ids[index]), '\n'

def get_batch_data_onehot_copyNet(X, y, batchid, batch_size, shuffle=True):
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
		
	batch_index = sample_index[batch_size*batchid:batch_size*(batchid+1)]
	X_batch = X[batch_index,:]
	y_batch = y[batch_index,:]
	print "round{0},batch size:{1}".format(batchid, X_batch.shape[0])
# 	X_decode_batch = y[batch_index,:]
# 	#delete all end flag in decode inputs
# 	X_decode_batch = np.where(X_decode_batch==cfg.end_flg_index, cfg.pad_flg_index, X_decode_batch)
# 	
# 	#delete all start flag in target sequences
# 	y_batch = np.hstack([y_batch[:,1:],np.ones((batch_index,1))*cfg.pad_flg_index])
	# reshape X to be [samples, time steps, features]
	X_decode_batch, y_batch = reshape_data_tf_copyNet(X_batch, y_batch)
# 	encoder_inputs_lengths = np.sum(X_batch!=cfg.pad_flg_index, axis=1)
# 	decoder_inputs_lengths = np.sum(X_decode_batch!=cfg.pad_flg_index, axis=1)
	return X_batch, X_decode_batch, y_batch

def restore_tf_model(model_prefix, sess, batch_size, is_train=True):
	# Load graph file.
	saver = tf.train.import_meta_graph(model_prefix + '.meta')
	saver.restore(sess, model_prefix)

	# Create model.
	model = model_maker.make_tf_tailored_seq2seq(
						n_encoder_layers = cfg.n_encoder_layers,
						n_decoder_layers = cfg.n_decoder_layers,
# 						dropout = cfg.ed_dropout,
						encoder_hidden_size=cfg.encoder_hidden_size, 
						decoder_hidden_size=cfg.decoder_hidden_size, 
						batch_size=batch_size, 
						embedding_dim=cfg.embedding_size, 
						vocab_size=cfg.vocab_size, 
						max_decode_iter_size=cfg.max_output_len,
						PAD = cfg.pad_flg_index,
						START = cfg.start_flg_index,
						EOS = cfg.end_flg_index,
						is_training = is_train,
						is_restored = True,
						)

	# Restore parameters in Model object with restored value.
	model.restore_from_session(sess)

	return model

# def decode_sequence_tf(X, model, save_path, beam_size=2):
# 	
# 	config = tf.ConfigProto()
# 	with tf.Session(config=config) as sess:
# 		restore_tf_model(sess, save_path)
# 		start_tokens = tf.ones([batch_size], dtype=tf.int32) * self._START
# 		beam_decoder = BeamSearchDecoder(
# 			beam_decoder_cell,
# 			word_embedding,
# 			start_tokens,
# 			self._EOS,
# 			tiled_decoder_initial_state,
# 			beam_width,
# 			output_layer=out_func,
# 		)
# 		
# 		sess.run([model.train_op, model.decoder_result_ids, model.loss, model._grads], feed_dict)
# 	return " ".join(decoded_sentence)
	
def train_tf_tailored_teaching_attention(
			sess, model, log_dir, ret_file_head, 
			X_train, Y_train, X_valid, Y_valid, 
			initial_epoch, batch_size=128, nb_epoch = 100):
	
	dataset = NumpySeqData(cfg.pad_flg_index, cfg.end_flg_index)
	dataset.load(X_train, Y_train, X_valid, Y_valid, cfg.vocab_i2word)
	dataset.build()
	
	data_process.dump(cfg.vocab_i2word, "../data/i2w.dic")
# 	np.savetxt("../data/X_train.txt", X_train.astype(np.int32), '%d')
# 	np.savetxt("../data/Y_train.txt", Y_train.astype(np.int32), '%d')
# 	np.savetxt("../data/X_valid.txt", X_valid.astype(np.int32), '%d')
# 	np.savetxt("../data/Y_valid.txt", Y_valid.astype(np.int32), '%d')
# 	num_train_examples = X_train.shape[0]
	max_epoch = nb_epoch
	step_nums = int(math.ceil(X_train.shape[0] / float(batch_size)))
	print "Total epoch:{0}, step num of each epoch:{1}".format(max_epoch, step_nums)
	
# 	steps_in_epoch = int(floor(num_train_examples / batch_size))

# 	model = Seq2SeqModel(config, input_batch=None)
# 	summary_op = model.summary_op

	saver = tf.train.Saver(max_to_keep=100)
# 	summary_writer = tf.train.summary.SummaryWriter('../log/tf/', sess.graph)
	train_summary_writer = tf.summary.FileWriter(log_dir, sess.graph_def)
	
	print "Training Start!"
	for epoch in range(initial_epoch, max_epoch):
		print "Epoch {}".format(epoch)
		epoch_loss = 0.
		epoch_acc = 0.
		epoch_acc_seq = 0.
		for step, data_dict in enumerate(dataset.train_datas(batch_size, False)):
			data_dict['keep_output_rate'] = cfg.ed_keep_rate
			data_dict['init_lr_rate'] = cfg.init_lr_rate
			data_dict['decay_step'] = cfg.decay_step
			data_dict['decay_factor'] = cfg.decay_factor
			feed_dict = model.make_feed_dict(data_dict)
# 			feed_dict['batch_size'] = data_dict['encoder_input'].shape[0]
			_, accuracy, accuracy_seqs, loss_value, grads, learn_rate, g_step, sampling_prob, summaries= \
			    sess.run([model.train_op, model.accuracy, model.accuracy_seqs, model.loss, model._grads, model.lr, model.train_step, model.sampling_probability, model.summary_op], feed_dict)

			if (step + 1) % 1 == 0:
				
				epoch_loss += loss_value
				epoch_acc += accuracy
				epoch_acc_seq += accuracy_seqs
				avg_loss = epoch_loss / (step + 1)
				avg_acc = epoch_acc / (step + 1)
				avg_acc_seq = epoch_acc_seq / (step + 1)

				print "Epoch:{epoch:3d}/{total_epoch:3d}, Step:{cur_step:6d}/{all_step:6d} ... Loss: {loss:.5f}/{loss_avg:.5f}, token_acc:{token_acc:.5f}/{acc_avg:.5f}, seq_acc:{seq_acc:.5f}/{seq_acc_avg:.5f}, grad:{grad:.8f}, sp:{sp:.8f}, lr:{lr:.8f}".format(
																			 epoch=epoch + 1,
																			 total_epoch=max_epoch,
																			 cur_step=step+1,
																			 all_step=step_nums,
																			 loss=loss_value,
			                    											 token_acc=accuracy,
			                    											 seq_acc=accuracy_seqs,
			                    											 loss_avg=avg_loss,
			                    											 acc_avg=avg_acc,
			                    											 seq_acc_avg=avg_acc_seq,
			                    											 grad = grads,
			                    											 sp = sampling_prob,
			                    											 lr=learn_rate)
			
# 				dataset.interpret_result(data_dict['encoder_inputs'], data_dict['decoder_inputs'], decoder_result_ids)

				train_summary_writer.add_summary(summaries, g_step)
		epoch_loss = epoch_loss / float(step_nums)
		epoch_acc = epoch_acc / float(step_nums)
		epoch_acc_seq = epoch_acc_seq / float(step_nums)
		right, wrong = 0.0, 0.0
		val_ret = []
		for step, data_dict in enumerate(dataset.val_datas(batch_size, False)):
			data_dict['keep_output_rate'] = cfg.de_keep_rate
			data_dict['init_lr_rate'] = cfg.init_lr_rate
			data_dict['decay_step'] = cfg.decay_step
			data_dict['decay_factor'] = cfg.decay_factor
			feed_dict = model.make_feed_dict(data_dict)
			beam_result_ids = sess.run(model.beam_search_result_ids, feed_dict)
			beam_result_ids = beam_result_ids[:, :, 0]
			print beam_result_ids.shape
			if step == 0:
				print(data_dict['decoder_inputs'][:5])
				print(beam_result_ids[:5])
			now_right, now_wrong, infos = dataset.eval_result(data_dict['encoder_inputs'], data_dict['decoder_inputs'], beam_result_ids, step, batch_size)
			right += now_right
			wrong += now_wrong
			val_ret.extend(infos)
# 		valid_summary_writer.add_summary(summaries_valid, epoch)
		path = "../checkpoints/tf/{prefix}.{epoch_id:02d}-{loss:.5f}-{acc:.5f}-{seq_acc:.5f}-{val_acc:.5f}.ckpt".format(prefix=ret_file_head,
																					    epoch_id=epoch,
																					    loss=epoch_loss,
																					    acc=epoch_acc,
																					    seq_acc=epoch_acc_seq,
																					    val_acc=100*right/float(right+wrong),
																					    )
		fp = open("../data/valid_ret_{}.txt".format(epoch), "w")
		for line in val_ret:
			fp.write('%s\n' % line)
		fp.close()
		data_process.extract_val_ret_err(epoch)
		saved_path = saver.save(sess, path, global_step=model.train_step)
		print "saved check file:" + saved_path
		print "Right: {}, Wrong: {}, Accuracy: {:.2}%".format(right, wrong, 100*right/float(right+wrong))
	
			
def train_tf_teaching_attention(model_fn, log_dir, ret_file_head, 
			X_train, Y_train, X_valid, Y_valid, 
			initial_epoch, batch_size=128, nb_epoch = 100):


	
	# placeholder for inputs
	p_encoder_inputs = tf.placeholder(tf.int32, shape=(batch_size, None))
	p_decoder_inputs = tf.placeholder(tf.int32, shape=(batch_size, None))
# 	p_decoder_outputs = tf.placeholder(tf.int32, shape=(batch_size, None))
	p_decoder_outputs = tf.placeholder(tf.int32, shape=(cfg.batch_size, None, cfg.vocab_size))
	
	# placeholder for sequence lengths
	p_encoder_inputs_lengths = tf.placeholder(tf.int32, shape=(batch_size,))
	p_decoder_inputs_lengths = tf.placeholder(tf.int32, shape=(batch_size,))
	p_decoder_outputs_lengths = tf.placeholder(tf.int32, shape=(batch_size,))
	p_embedding_placeholder = tf.placeholder(tf.float32, [cfg.vocab_size, cfg.embedding_size])
	embedding = np.random.randn(cfg.vocab_size, cfg.embedding_size)
	
	args_input = {}
	args_input['batch_size'] = batch_size
	args_input['encoder_inputs'] = p_encoder_inputs
	args_input['decoder_inputs'] = p_decoder_inputs
	args_input['decoder_outputs'] = p_decoder_outputs
	args_input['encoder_inputs_lengths'] = p_encoder_inputs_lengths
	args_input['decoder_inputs_lengths'] = p_decoder_inputs_lengths
	args_input['decoder_outputs_lengths'] = p_decoder_outputs_lengths
	args_input['embedding_placeholder'] = p_embedding_placeholder
	args_input['embedding'] = embedding
	max_batchid = int(np.ceil(X_train.shape[0] / batch_size))

	with tf.variable_scope('root'):
		train_loss, train_op, embedding_init = model_fn(args_input, mode='train')
	
	with tf.variable_scope('root', reuse=True):
		eval_loss = model_fn(args_input, mode='eval')
	
	with tf.variable_scope('root', reuse=True):
		translations = model_fn(args_input, mode='infer')
	
	# session configure for GPU
	sess_config = tf.ConfigProto()
	# sess_config.gpu_options.allow_growth = True
	# sess_config.per_process_gpu_memory_fraction = 0.5
	with tf.Session(config=sess_config) as sess:
		saver = tf.train.Saver()
# 		sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# 		sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
		# merge all summaries so far and initialize a FileWriter
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)
		
		sess.run(tf.global_variables_initializer())
		sess.run(embedding_init, feed_dict={p_embedding_placeholder: embedding})
		for epoch in range(nb_epoch): 
			for batchid in range(max_batchid):
				# training
	# 			train_batch = train_data.next_batch()
				encoder_inputs, decoder_inputs, decoder_outputs = get_batch_data_tf_teaching(X_train, Y_train, 
																							batchid, batch_size, reshape_fn=reshape_data_tf_teach_onehot_y)
				feed_dict = {
				    p_encoder_inputs: encoder_inputs,
				    p_decoder_inputs: decoder_inputs,
				    p_decoder_outputs: decoder_outputs,
	# 				p_encoder_inputs_lengths: np.ones((batch_size,)) * cfg.max_input_len,
	# 				p_decoder_inputs_lengths: np.ones((batch_size,)) * cfg.max_output_len
					p_encoder_inputs_lengths: np.sum(encoder_inputs!=cfg.pad_flg_index, axis=1),
					p_decoder_inputs_lengths: np.sum(decoder_inputs!=cfg.pad_flg_index, axis=1),
	# 				p_decoder_outputs_lengths: np.sum(decoder_inputs!=cfg.pad_flg_index, axis=1),
					p_decoder_outputs_lengths: np.ones((batch_size,)) * cfg.max_output_len,
				} 
				loss_train, _ = sess.run([train_loss, train_op], feed_dict=feed_dict)
 				print 'batchid = %d, train loss = %f' % (batchid, loss_train)
# 				summary = sess.run(merged)
# 				writer.add_summary(summary, epoch * max_batchid + batchid)
# 				print('epoch {:d}, batch {:d}, cross-entropy {:.5f}'.format(epoch+1, batchid+1, loss_train))
				# save checkpoints
				if (batchid + 1) % cfg.save_freq == 0:
					saver.save(sess, '../checkpoints/', global_step=batchid)
					print 'model saved'
				
	 			# eval
	 			if (batchid + 1) % 10 == 0:
	 				encoder_inputs = X_valid
	 				decoder_inputs, decoder_inputs, decoder_outputs = reshape_data_tf_teach_onehot_y(X_valid, Y_valid)
	 				feed_dict = {
	 				    p_encoder_inputs: encoder_inputs,
	 				    p_decoder_inputs: decoder_inputs,
	 				   	p_decoder_outputs: decoder_outputs,
	 				    p_encoder_inputs_lengths: np.sum(encoder_inputs!=cfg.pad_flg_index, axis=1),
	 				    p_decoder_inputs_lengths: np.sum(decoder_inputs!=cfg.pad_flg_index, axis=1),
	 				    p_decoder_outputs_lengths: np.ones((batch_size,)) * cfg.max_output_len,
	 				}
	 				loss_eval = sess.run(eval_loss, feed_dict=feed_dict)
	 				print '\t\tbatchid = %d, eval loss = %f' % (batchid, loss_eval)
				
	 			# inference
	 			if (batchid + 1) % 10 == 0:
	 				encoder_inputs, decoder_inputs, decoder_outputs = reshape_data_tf_teach_onehot_y(X_valid, Y_valid)
	 				feed_dict = {
	 				    encoder_inputs: encoder_inputs,
	 				    decoder_inputs: decoder_inputs,
	 				    decoder_outputs: decoder_outputs,
	 				    p_encoder_inputs_lengths: np.sum(encoder_inputs!=cfg.pad_flg_index, axis=1),
	 				    p_decoder_inputs_lengths: np.sum(decoder_inputs!=cfg.pad_flg_index, axis=1),
	 				    p_decoder_outputs_lengths: np.ones((batch_size,)) * cfg.max_output_len,
	 				}
	 				word_ids = sess.run(translations, feed_dict=feed_dict)
	 				print 'batchid = %d' % (batchid)
	 				index = random.randint(0, 39)
	
	 				print 'input:', cfg.is2sentence(encoder_inputs[index]), '\n'
# 	 				print 'output:', vocab.word_ids_to_sentence(
# 	 				        train_batch.decoder_inputs[index]), '\n'
	 				print 'predict:', cfg.is2sentence(word_ids[index]), '\n'

def train_onehot(model, log_dir, ret_file_head, X_train, Y_train, X_valid, Y_valid, initial_epoch, batch_size=128, nb_epoch = 100):

	# reshape X to be [samples, time steps, features]
# 	X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
# 	Y_valid = np.reshape(Y_valid, (Y_valid.shape[0], Y_valid.shape[1], 1))
	X_valid, Y_valid = reshape_data_onehot(X_valid, Y_valid)
# 	print X_train.getnnz()
# 	print X_train.shape[0]
# 	print X_train.shape[1]

	board = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head)
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=False)
	# start training
	start_time = time.time()
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
#  	samples_per_epoch = batch_size
	model.fit_generator(generator=generator_onehot(X_train, Y_train, batch_size, False), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
			    	    validation_data=(X_valid, Y_valid),
			    		callbacks=[board, checkpointer], 
			    		initial_epoch=initial_epoch)
	print 'Training time', time.time() - start_time
	# evaluate network
# 	decode_sequence_teach(X_valid, Y_valid, extend_args)
	
	score = model.evaluate(X_valid, Y_valid, batch_size)
	p_y = model.predict(X_valid, batch_size, verbose=0)
	out_list = get_predict_list(p_y)
	out_list = format_out_list(out_list)
# 	output_to_csv(out_list, df_valid, path + ret_file_head + '.csv')
	print out_list
# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)



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
	predict_y = np.argmax(predict_y, axis=2)
	out_list = arrs_to_sens(predict_y)

	return out_list


	
	

def gen_model_2(depth=1, dropout=0.3):
	model = model_maker.make_AttentionSeq2Seq(input_dim=cfg.input_vocab_size, 
							input_length=cfg.max_input_len, 
							hidden_dim=cfg.input_hidden_dim, 
							output_length=cfg.max_output_len, 
							output_dim=cfg.output_vocab_size, 
							dropout = dropout,
							depth = depth)
	model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01), metrics=['accuracy'])
	return model


def gen_model_3(depth=1, dropout=0.3, peek=True, teacher_force=True):
	model = model_maker.make_Seq2Seq(batch_input_shape=(None, cfg.max_input_len, cfg.input_vocab_size),
						 hidden_dim=cfg.input_vocab_size, 
					 	output_length=cfg.max_output_len,
					  	output_dim=cfg.output_vocab_size, 
					  	depth=depth, teacher_force=teacher_force, dropout=dropout,
					 	peek=peek)
	model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
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
			
def gen_model_teaching_LSTM():
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
	model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
# 	extend_args = {'encoder_inputs':encoder_inputs, "encoder_states":encoder_states, "decoder_inputs":decoder_inputs,
# 				   'decoder_lstm':decoder_lstm, 'decoder_dense':decoder_dense }
	return model

# def attention_3d_block(inputs):
# 	# inputs.shape = (batch_size, time_steps, input_dim)
# 	input_dim = int(inputs.shape[2])
# 	a = Permute((2, 1))(inputs)
# 	a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
# 	a = Dense(TIME_STEPS, activation='softmax')(a)
# 	if SINGLE_ATTENTION_VECTOR:
# 		a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
# 		a = RepeatVector(input_dim)(a)
# 	a_probs = Permute((2, 1), name='attention_vec')(a)
# 	output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
# 	return output_attention_mul
# 
# 
# def model_attention_applied_after_lstm():
# 	inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
# 	lstm_units = 32
# 	lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
# 	attention_mul = attention_3d_block(lstm_out)
# 	attention_mul = Flatten()(attention_mul)
# 	output = Dense(1, activation='sigmoid')(attention_mul)
# 	model = Model(input=[inputs], output=output)
# 	return model

def gen_model_teaching_attention():

	# Define an input sequence and process it.
 	encoder_inputs = Input(shape=(cfg.max_input_len, cfg.input_vocab_size))
#  	input_vec_en = embedding_layer_en(encoder_inputs)
#  	reshaper_en = Reshape((cfg.max_input_len, embedding_len))
#  	input_vec_en = reshaper_en(input_vec_en)
 	
	encoder = LSTM(cfg.input_vocab_size, return_state=True, return_sequences=True)
	state_out, state_h, state_c = encoder(encoder_inputs)
	
	attention_layer_o = Dense(cfg.max_input_len, activation='softmax')
	out = Permute((2, 1))(state_out)
	out = Reshape((cfg.input_vocab_size, cfg.max_input_len))(out)
	out = attention_layer_o(out)
	w_state_o = Permute((2, 1), name='attention_vec')(out)
	output_attention_o = merge([state_out, w_state_o], name='attention_mul', mode='mul')
	
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]
	
	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(cfg.max_output_len, cfg.output_vocab_size))
	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the
	# return states in the training model, but we will use them in inference.
	
	decoder_lstm = LSTM(cfg.input_vocab_size, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm([output_attention_o, decoder_inputs], initial_state=encoder_states)
	decoder_dense = Dense(cfg.output_vocab_size, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	
	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)



# 	model = model_maker.make_attentionSeq2Seq(input_dim=cfg.input_vocab_size, 
# 							input_length=cfg.max_input_len, 
# 							hidden_dim=cfg.input_vocab_size, 
# 							output_length=cfg.max_output_len, 
# 							output_dim=cfg.output_vocab_size, 
# 							bidirectional=False,
# 							dropout = 0.0,
# 							depth = 1)
	
# 	model = model_maker.make_Seq2Seq(batch_input_shape=(None, cfg.max_input_len, cfg.input_vocab_size),
# 						 hidden_dim=cfg.input_vocab_size, 
# 					 	output_length=cfg.max_output_len,
# 					  	output_dim=cfg.output_vocab_size, 
# 					  	depth=1, teacher_force=True,
# 					 	peek=True)
	
	model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
	
	
	
	
	return model

def gen_model_teaching_GRU():
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
 	
	encoder = GRU(cfg.input_vocab_size, return_state=True)
	state_h, state_c = encoder(encoder_inputs)
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = state_h
	
	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(cfg.max_output_len, cfg.output_vocab_size))
	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the
	# return states in the training model, but we will use them in inference.
# 	input_vec_de = embedding_layer_de(decoder_inputs)
# 	reshaper_de = Reshape((cfg.max_output_len, embedding_len))
# 	input_vec_de = reshaper_de(input_vec_de)
	
	decoder_lstm = GRU(cfg.input_vocab_size, return_sequences=True, return_state=True)
	decoder_outputs, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
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
	
def gen_model_lstm():
	model = Sequential()
	model.add(Masking(mask_value=0, input_shape=(cfg.max_input_len, cfg.vocab_size)))
	model.add(LSTM(cfg.input_vocab_size, implementation = 2, input_shape=(cfg.max_input_len, cfg.vocab_size)))
	model.add(RepeatVector(cfg.max_output_len))
	model.add(LSTM(cfg.input_vocab_size, return_sequences=True, implementation = 2))
	model.add(TimeDistributed(Dense(cfg.vocab_size, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
	return model

def gen_model_gru():
	model = Sequential()
	model.add(Masking(mask_value=0, input_shape=(cfg.max_input_len, cfg.input_vocab_size)))
	model.add(GRU(cfg.encoder_hidden_size, implementation = 2, input_shape=(cfg.max_input_len, cfg.input_vocab_size)))
	model.add(RepeatVector(cfg.max_output_len))
	model.add(GRU(cfg.decoder_hidden_size, return_sequences=True, implementation = 2))
	model.add(TimeDistributed(Dense(cfg.vocab_size, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
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
	

	
def experiment_simple_lstm(nb_epoch=100, input_num=0, cls_id=1, pre_train_model_file=None):
	data = np.load("../data/train_cls{}.npz".format(cls_id))
	x_t_c = data['x_t_c']
	y_t = data['y_t']
	if input_num > 0:
		x_t_c = x_t_c[:input_num]
		y_t = y_t[:input_num]
	x_train, x_valid, y_train, y_valid = train_test_split(x_t_c, y_t, test_size=1000, random_state=0)
	model = gen_model_lstm()
	print(model.summary())
	
	initial_epoch = 0
	#load pre-training weight
	if pre_train_model_file is not None:
		print "Load pre-training weight from file:" + pre_train_model_file
		model.load_weights(pre_train_model_file)
		index_start = pre_train_model_file.find("_weights.") + len('_weights.')
 		index_end = pre_train_model_file.find("-")
		initial_epoch = int(pre_train_model_file[index_start:index_end])
	log_dir = '../logs/nt{}/'.format(cls_id)
# 	model = AttentionSeq2Seq(input_dim=cfg.max_input_len, hidden_dim=cfg.input_hidden_dim, output_length=cfg.max_output_len, output_dim=cfg.output_vocab_size, depth=2)
# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
# 	print(model.summary())
	train_onehot(model, log_dir, "l1_l1_c" + str(cls_id), x_train, y_train, x_valid, y_valid, initial_epoch,
					batch_size=256, nb_epoch = nb_epoch + initial_epoch)

def experiment_simple_gru(nb_epoch=100, input_num=0, cls_id=1, pre_train_model_file=None):
	data = np.load("../data/train_cls{}.npz".format(cls_id))
	x_t_c = data['x_t_c']
	y_t = data['y_t']
	if input_num > 0:
		x_t_c = x_t_c[:input_num]
		y_t = y_t[:input_num]
	x_train, x_valid, y_train, y_valid = train_test_split(x_t_c, y_t, test_size=1000, random_state=0)
	model = gen_model_gru()
	print(model.summary())
	
	initial_epoch = 0
	#load pre-training weight
	if pre_train_model_file is not None:
		print "Load pre-training weight from file:" + pre_train_model_file
		model.load_weights(pre_train_model_file)
		index_start = pre_train_model_file.find("_weights.") + len('_weights.')
 		index_end = pre_train_model_file.find("-")
		initial_epoch = int(pre_train_model_file[index_start:index_end])
	log_dir = '../logs/nt{}/'.format(cls_id)
# 	model = AttentionSeq2Seq(input_dim=cfg.max_input_len, hidden_dim=cfg.input_hidden_dim, output_length=cfg.max_output_len, output_dim=cfg.output_vocab_size, depth=2)
# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
# 	print(model.summary())
	train_onehot(model, log_dir, "g1_g1_c" + str(cls_id), x_train, y_train, x_valid, y_valid, initial_epoch,
					batch_size=256, nb_epoch = nb_epoch + initial_epoch)
	
def experiment_simple1_bi():
	x_train, y_train, x_valid, y_valid, _, df_valid = data_process.gen_data_from_npz("../data/train_filted.npz")
	x_t = sparse.csr_matrix(x_train)
# 	x_v = sparse.csr_matrix(x_valid)
	x_v = x_valid
	
	model = gen_model_gru()
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
	
def experiment_attention1(input_num=0, cls_id=1, pre_train_model_file=None):
	data = np.load("../data/train_cls{}.npz".format(cls_id))
	x_t_c = data['x_t_c']
	y_t = data['y_t']
	if input_num > 0:
		x_t_c = x_t_c[:input_num]
		y_t = y_t[:input_num]
	x_train, x_valid, y_train, y_valid = train_test_split(x_t_c, y_t, test_size=1000, random_state=0)
	model = gen_model_2()
	print(model.summary())
	
	initial_epoch = 0
	#load pre-training weight
	if pre_train_model_file is not None:
		print "Load pre-training weight from file:" + pre_train_model_file
		model.load_weights(pre_train_model_file)
		index_start = pre_train_model_file.find("_weights.") + len('_weights.')
 		index_end = pre_train_model_file.find("-")
		initial_epoch = int(pre_train_model_file[index_start:index_end])
	log_dir = '../logs/nt{}/'.format(cls_id)
# 	model = AttentionSeq2Seq(input_dim=cfg.max_input_len, hidden_dim=cfg.input_hidden_dim, output_length=cfg.max_output_len, output_dim=cfg.output_vocab_size, depth=2)
# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
# 	print(model.summary())
	train_onehot(model, log_dir, "attention_l1_l1_c" + str(cls_id), x_train, y_train, x_valid, y_valid, initial_epoch,
					batch_size=100, nb_epoch = 100 + initial_epoch)
	
	
# def load_tf_model(model_fn, model_file):
# 	print "Load pre-training weight from file:" + model_file
# 	model_fn.load_weights(model_file)
# 	index_start = model_file.find("_weights.") + len('_weights.')
# 	index_end = model_file.find("-")
# 	initial_epoch = int(model_file[index_start:index_end])

	
		
def experiment_teaching_tf(batch_size=256, nb_epoch=100, input_num=0, test_size=100000, cls_id=0, file_head="tf_teach_att_bl2_bl1_c", is_debug=False, pre_train_model_prefix=None):
# 	data = np.load("../data/train_cls{}.npz".format(cls_id))
# 	x_t_c = data['x_t_c']
# 	y_t = data['y_t']

	x_train, y_train = data_process.get_training_data_from_files("../data/train/")
	x_t_c_v, y_t_v = data_process.get_training_data_from_files("../data/valid/")

	if input_num > 0:
		x_train = x_train[:input_num]
		y_train = y_train[:input_num]
	_, x_valid, _, y_valid = train_test_split(x_t_c_v, y_t_v, test_size=test_size, random_state=0)

	print "train items num:{0}, valid items num:{1}".format(x_train.shape[0], x_valid.shape[0])

	with tf.Session() as sess:
		# Create model or load pre-trained model.
		if is_debug:
			sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		model = None
		if pre_train_model_prefix is None:
			model = model_maker.make_tf_tailored_seq2seq(
					n_encoder_layers = cfg.n_encoder_layers,
					n_decoder_layers = cfg.n_decoder_layers,
# 					dropout = cfg.ed_dropout,
					encoder_hidden_size=cfg.encoder_hidden_size, 
					decoder_hidden_size=cfg.decoder_hidden_size, 
					batch_size=batch_size, 
					embedding_dim=cfg.embedding_size, 
					vocab_size=cfg.vocab_size, 
					max_decode_iter_size=cfg.max_output_len,
					PAD = cfg.pad_flg_index,
					START = cfg.start_flg_index,
					EOS = cfg.end_flg_index,
					)
			initial_epoch = 0
			sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		else:
			m = re.match('.*_c\d+\.(\d+)\-.*', pre_train_model_prefix)
			assert (m and len(m.groups()) > 0), 'Failed to get epoch number while restoring model!'
			initial_epoch = int(m.group(1))
			model = restore_tf_model(pre_train_model_prefix, sess, batch_size)

		log_dir = '../logs/tf/'
# 	model = AttentionSeq2Seq(input_dim=cfg.max_input_len, hidden_dim=cfg.input_hidden_dim, output_length=cfg.max_output_len, output_dim=cfg.output_vocab_size, depth=2)
# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
# 	print(model.summary())
		train_tf_tailored_teaching_attention(sess, model, log_dir, file_head + str(cls_id), x_train, y_train, x_valid, y_valid, initial_epoch,
				batch_size=batch_size, nb_epoch = nb_epoch + initial_epoch)


	
def experiment_teaching1(nb_epoch=100, input_num=0, cls_id=1, pre_train_model_file=None):
	data = np.load("../data/train_cls{}.npz".format(cls_id))
	x_t_c = data['x_t_c']
	y_t = data['y_t']
	if input_num > 0:
		x_t_c = x_t_c[:input_num]
		y_t = y_t[:input_num]
	x_train, x_valid, y_train, y_valid = train_test_split(x_t_c, y_t, test_size=10000, random_state=0)
	model = gen_model_teaching_LSTM()
	print(model.summary())
	initial_epoch = 0
	#load pre-training weight
	if pre_train_model_file is not None:
		print "Load pre-training weight from file:" + pre_train_model_file
		model.load_weights(pre_train_model_file)
		index_start = pre_train_model_file.find("_weights.") + len('_weights.')
 		index_end = pre_train_model_file.find("-")
		initial_epoch = int(pre_train_model_file[index_start:index_end])
	log_dir = '../logs/nt{}/'.format(cls_id)
# 	model = AttentionSeq2Seq(input_dim=cfg.max_input_len, hidden_dim=cfg.input_hidden_dim, output_length=cfg.max_output_len, output_dim=cfg.output_vocab_size, depth=2)
# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
# 	print(model.summary())
	train_teaching_onehot(model, log_dir, "teach_g1_g1_c" + str(cls_id), x_train, y_train, x_valid, y_valid, initial_epoch,
					batch_size=256, nb_epoch = nb_epoch + initial_epoch)
	
	
	
def experiment_teaching0(nb_epoch=100, input_num=0, add_index_file="../data/en_train_test_ret_err_index", add_cnt=10, cls_id=0, pre_train_model_file=None):
	data = np.load("../data/en_train.npz".format(cls_id))
	x_t_c = data['x_t_c']
	y_t = data['y_t']
	if input_num > 0:
		x_t_c = x_t_c[:input_num]
		y_t = y_t[:input_num]
	if add_index_file is not None:
		indies = data_process.load(add_index_file)
		print "add additional error training data:" + str(len(indies))
		add_x = x_t_c[indies]
		add_y = y_t[indies]
		for i in range(add_cnt):
			x_t_c = np.vstack([x_t_c, add_x])
			y_t = np.vstack([y_t, add_y])
	
	x_train, x_valid, y_train, y_valid = train_test_split(x_t_c, y_t, test_size=10000, random_state=0)
	print "class id {0}, total training data {1}, valid data{2}".format(cls_id, x_train.shape[0], x_valid.shape[0])
	
	model = gen_model_teaching_LSTM()
	print(model.summary())
	
	initial_epoch = 0
	#load pre-training weight
	if pre_train_model_file is not None:
		print "Load pre-training weight from file:" + pre_train_model_file
		model.load_weights(pre_train_model_file)
		index_start = pre_train_model_file.find("_weights.") + len('_weights.')
 		index_end = pre_train_model_file.find("-")
		initial_epoch = int(pre_train_model_file[index_start:index_end])
	log_dir = '../logs/nt{}/'.format(cls_id)
# 	model = AttentionSeq2Seq(input_dim=cfg.max_input_len, hidden_dim=cfg.input_hidden_dim, output_length=cfg.max_output_len, output_dim=cfg.output_vocab_size, depth=2)
# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
# 	print(model.summary())
	
	train_teaching_onehot(model, log_dir, "teach_l1_l1_c" + str(cls_id), x_train, y_train, x_valid, y_valid, initial_epoch,
					batch_size=256, nb_epoch = nb_epoch + initial_epoch)	
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
	experiment_simple_gru()
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

def split_by_classifier(classifier, cls_num, x_t_cls, x_t, df_test, y_t_cls=None, y_t=None):
	index_list = []
	x_list = []
	cls_df_list = []
	cls_y_list = []
	y_list = []
	if classifier is None:
		index_list = [range(len(df_test))]
		x_list = [x_t]
		cls_df_list = [df_test]
		cls_y_list = [y_t_cls]
		y_list = [y_t]
		return index_list, x_list, cls_df_list, cls_y_list, y_list
	
	predict_y = classifier.predict(x_t_cls, batch_size = 256, verbose=0)
	p_y = np.argmax(predict_y, axis=1)

	for cls in range(cls_num):
		cls_id = cls + 1
		index = np.where(p_y==cls)[0].tolist()
		index_list.append(index)
		x_list.append(x_t[index])
		cls_df_list.append(df_test.iloc[index].reset_index(drop=True))
		cls_df_list[cls]['p_new_class'] = cls_id
		if y_t_cls is not None:
			cls_y_list.append(y_t_cls[index])
		if y_t is not None:
			y_list.append(y_t[index])

	
	return index_list, x_list, cls_df_list, cls_y_list, y_list


def run_normalize(is_evaluate = False, test_size=0, use_classifier = True, data_args=cfg.data_args_train, is_teach=False):
	df_test = None
	if test_size > 0:
		df_test = pd.read_csv(data_args['df_test'], nrows=test_size)
	else:
		df_test = pd.read_csv(data_args['df_test'])
		
	df_test['len'] = df_test['before'].apply(lambda x:len(str(x)))
	cfg.max_output_len_decode = df_test['len'].max()
	print "Max decode_teach output length is {}".format(cfg.max_output_len_decode)
	del df_test['len']

	data = np.load(data_args['feat_classify'])
	x_t_cls = data['x_t']
	data = np.load(data_args['feat_normalization'])
# 	x_t = data['x_t']
	x_t_c = data['x_t_c']
	np.savetxt("../data/X_test.txt", x_t_c.astype(np.int32), '%d')
	
	if test_size > 0:
		x_t_c = x_t_c[:test_size]
		x_t_cls = x_t_cls[:test_size]
		
	cls_num = 1
	mod_classifier = None
	if use_classifier:
		mod_classifier = gen_model_class_2()
		print "Load classify model..."
		print mod_classifier.summary()
		mod_classifier.load_weights(data_args['model_classify'])
		cls_num = len(data_args['model_normal'])
	index_list, x_list, cls_df_list, _, _ = split_by_classifier(mod_classifier, cls_num, x_t_c, x_t_c, df_test)
	
	cls_info = "Total item num:{}, ".format(len(df_test))

	for cls in range(cls_num):
		cls_info = cls_info + 'c{0} num:{1} '.format(cls + 1, len(cls_df_list[cls]))
		if is_teach:
			decode_teach(x_list[cls], cls_df_list[cls], weights_file=data_args['model_normal'][cls])
		else:
			#decode(x_list[cls], cls_df_list[cls], weights_file=data_args['model_normal'][cls])
			decode_tf(x_list[cls], cls_df_list[cls], data_args['model_normal'][cls])
		if is_evaluate:
			evalute_result(x_list[cls], "../data/noraml_err_cls{}.csv".format(cls + 1))
	
	print cls_info
		
	gen_result_file(index_list, df_test, cls_df_list, 
					test_ret_file=data_args['test_ret_file'], ret_file=data_args['ret_file'], origin_file=data_args['origin_file'],
					sub_file=data_args['sub_file'], test_ret_file_err=data_args['test_ret_file_err'])
	
	
def gen_result_file(index_list, df_test, cls_df_list, test_ret_file, ret_file, origin_file, sub_file=None, test_ret_file_err=None):
	df_origin = pd.read_csv(origin_file)
	df_origin['p_after'] = df_origin['before']

	is_classified = (len(index_list) > 1)
	if is_classified:
		df_test['p_after'] = ''
		for cls in range(len(index_list)):
			df_test.at[index_list[cls], 'p_after'] = cls_df_list[cls]['p_after'].values.tolist()
	else:
		df_test = cls_df_list[0]

# 	df_origin['id'] = df_origin.apply(lambda x: str(x['sentence_id']) + "_" + str(x['token_id']), axis=1)
# 	df_test['id'] = df_test.apply(lambda x: str(x['sentence_id']) + "_" + str(x['token_id']), axis=1)
	index_origin = np.where(df_origin['new_class'].values != 0)[0].tolist()
	print len(df_origin)
	print len(index_origin)
	print len(df_test)
	
	
# 	test_list = map(lambda x : x + "\n", test_list)
# 	f = open("../data/test_temp.txt", 'w')
# 	f.writelines(test_list)
# 	f.close()
	#only when origin data length equal to the test data length, we need to cope with it, otherwise, skip.
	if len(index_origin) == len(df_test):
		df_origin.at[index_origin, 'p_after'] = df_test['p_after'].values.tolist()
		#if it is copy flag, then copy the before value
		#df_origin['p_after'] = df_origin.apply(lambda x:x['before'] if str(x['p_after']) == cfg.unk_flg else x['p_after'], axis=1)
		df_origin.to_csv(ret_file, index=False)
		print "saved predicted origin file:" + ret_file

		if sub_file is not None:
			df_sub = pd.DataFrame()
			df_sub['id'] = df_origin.apply(lambda x: '%s_%s' % (str(x['sentence_id']), str(x['token_id'])), axis=1)
			df_sub['after'] = df_origin['p_after']
			df_sub.to_csv(sub_file, quoting=csv.QUOTE_ALL, index=False)
			print 'saved submission file:' + sub_file
	else:
		test_list = df_test['p_after'].values.tolist()
	
# 		test_list = map(lambda x : x + "\n", test_list)
# 		f = open("../data/test_temp.txt", 'w')
# 		f.writelines(test_list)
# 		f.close()
# 		print df_origin.loc[index_origin[0:len(test_list)]]
# 		print index_origin[0:len(test_list)]
		df_origin.at[index_origin[0:len(test_list)], 'p_after'] = test_list
		df_origin = df_origin.loc[index_origin[0:len(test_list)]]
		df_origin.to_csv("../data/test_temp.csv", index=False)
		print "test data not equal to the origin data, skip submission stage. output ../data/test_temp.csv"
		
	df_test['p_after'] = df_test.apply(lambda x:x['before'] if str(x['p_after']).strip() == cfg.unk_flg else x['p_after'], axis=1)
	
# 	for i in range(len(df_test)):
# 		p_after = df_test.at[i, 'p_after']
# 		if str(p_after).strip() == cfg.unk_flg:
# 			df_test.at[i, 'p_after'] = df_test.at[i, 'before']
	#if has after field, we can calculate accuracy for each class
	#switch column order
	p_after = df_test['p_after']
	df_test.drop(labels=['p_after'], axis=1,inplace = True)
	df_test.insert(5, 'p_after', p_after)
	
	if 'after' in df_test.columns.values.tolist():
		info = "Total noramlization acc:{}, ".format(len(df_test[df_test['p_after']==df_test['after']])/float(len(df_test)))
		if is_classified:
			for cls in range(len(index_list)):
				cls_id = cls + 1
				cls_num = len(index_list[cls])
				corrected_cls_num = len(df_test[(df_test['new_class1']==cls_id) & (df_test['p_after']==df_test['after'])])
				acc = corrected_cls_num/float(cls_num)
				info = info + "cls{0} acc:{1}   ".format(cls_id, acc)
		print info
		df_test_err = df_test[df_test['p_after']!=df_test['after']]
		if test_ret_file_err is not None:
			print "normalization error number is:" + str(len(df_test_err))
			print df_test_err.head()
			df_test_err.to_csv(test_ret_file_err)
			print 'saved test normalization error result file:' + test_ret_file_err
			err_index_file = test_ret_file_err[:test_ret_file_err.find(".csv")] + "_index"
			data_process.dump(df_test_err.index.tolist(), err_index_file)
			print "saved error index to file:" + err_index_file
	
	df_test.to_csv(test_ret_file, index=False)	
	print 'saved test result file:' + test_ret_file
	

		
def evalute_result(df_ret, err_file):
	total = len(df_ret)
	df_corrected = df_ret[df_ret['after']==df_ret['p_after']]
	df_err = df_ret[df_ret['after']!=df_ret['p_after']]
	print "decoded token num:%d, err num:%d, acc:%f"%(total, len(df_err), df_corrected/float(total))
	df_err.to_csv(err_file, index=False)

def decode(X, df_test, weights_file):
	model = gen_model_gru()
	model.load_weights(weights_file)
	print "Load Normalization Model..."
	print(model.summary())
	decoder = model
	N = len(df_test)
	normalizations = []
	for i in range(N):
		before = str(df_test.at[i, 'before'])
		after = decode_sequence(decoder, X[i])
		normalizations.append(after)
		print "sentence%d:%s -> %s"%(i, before, after)
		
	df_test['p_after'] = pd.Series(normalizations)
	
def decode_tf(X, df_test, model_prefix, batch_size=256):
	def calculate_lens(data):
		len_list = []
		PAD = 0
		for i in range(data.shape[0]):
			tmp_list = data[i].tolist()
			pos = len(tmp_list)
			if PAD in tmp_list:
				pos = tmp_list.index(PAD)
			len_list.append(pos)

		return np.asarray(len_list, dtype=np.int32)

	with tf.Session() as sess:
		# Restore trained model.
		model = restore_tf_model(model_prefix, sess, 256, False)

		# Do normalization on test data.
		normalizations = []
		total_case = X.shape[0]
		total_batch = int(math.ceil(total_case / float(batch_size)))
		for i in range(total_batch):
			print '---> Finished batch : %d/%d' % (i, total_batch)
			begin_idx = i * batch_size
			size = min(batch_size, total_case - begin_idx)
			end_idx = begin_idx + size
			batch_X = X[begin_idx : end_idx]
			batch_X_lens = calculate_lens(batch_X)
			Y_empty = np.asarray([[]] * size)
			Y_empty_lens = np.zeros(size)
			
			data_dict={}
			data_dict['encoder_inputs'] = batch_X
			data_dict['encoder_lengths'] = batch_X_lens
			data_dict['decoder_inputs'] = Y_empty
			data_dict['decoder_lengths'] = Y_empty_lens
			data_dict['keep_output_rate'] = cfg.de_keep_rate
			data_dict['init_lr_rate'] = cfg.init_lr_rate
			data_dict['decay_step'] = cfg.decay_step
			data_dict['decay_factor'] = cfg.decay_factor
			feed_dict_de = model.make_feed_dict(data_dict)
			# Run prediction on batch data.
			beam_result_ids = sess.run(
					model.beam_search_result_ids,
					feed_dict = feed_dict_de)
				
			beam_result_ids = beam_result_ids[:, :, 0]

			# Tranform output to real tokens.
			for j in range(size):
				before = str(df_test.at[begin_idx + j, 'before'])
				ids = beam_result_ids[j].tolist()
				p_after = data_process.recover_y_info(ids)
				normalizations.append(p_after)
				#print '### %d' % (begin_idx + j), before, '-->', p_after

	# Save to column of dataframe.
	df_test['p_after'] = pd.Series(normalizations)

	# Post process of the result output by model.
	df_test['p_after'] = df_test.apply(lambda x:x['before'] if str(x['p_after']) == cfg.unk_flg else x['p_after'], axis=1)
	df_temp = pd.DataFrame(columns=['changed', 'before', 'p_after', 'norm_after'])
	for i in range(len(df_test)):
		before = unicode(str(df_test.loc[i, 'before']))
		p_after = unicode(str(df_test.loc[i, 'p_after']))
		normed, norm_after = rule_norm_obj.normalize(before)
		if normed:
			if norm_after != p_after:
				df_test.at[i, 'p_after'] = norm_after
				df_temp.loc[len(df_temp)] = ['1', before, p_after, norm_after]
			else:
				df_temp.loc[len(df_temp)] = ['0', before, p_after, norm_after]
# 	df_temp.to_csv('debug.csv', index=False)


def decode_teach(X, df_test, weights_file):
	
	model = gen_model_teaching_LSTM()
	model.load_weights(weights_file)
	print "Load Normalization Model..."
	print(model.summary())
	
	encoder, decoder = load_decode_model(model)
	N = len(df_test)
	normalizations = []
	for i in range(N):
		before = str(df_test.at[i, 'before'])
		after = decode_sequence_teach(encoder, decoder, X[i])
		normalizations.append(after)
		print "sentence%d:%s -> %s"%(i, before, after)
		
	df_test['p_after'] = pd.Series(normalizations)
	
def load_decode_model(model):

	encoder_weight = model.get_layer(index=2).get_weights()
	decoder_weight = model.get_layer(index=3).get_weights()
	dense_weight = model.get_layer(index=4).get_weights()
	
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
	
	encoder.set_weights(encoder_weight)
	decoder_lstm.set_weights(decoder_weight)
	decoder_dense.set_weights(dense_weight)
	return encoder_model, decoder_model

def decode_sequence(decoder, X, beam_size=2):
	
	#reshape X to one hot matrix
	x = np.zeros((1, X.shape[0], cfg.input_vocab_size))
	for t in range(X.shape[0]):
		x[0, t, X[t]] = 1.0
	
	# Encode the input as state vectors.
# 	print states_value[0].shape
# 	print states_value[1].shape
	decoded_sentence =  search_path_maxone(x, decoder=decoder, beam_size=beam_size)
	
	return " ".join(decoded_sentence)

def search_path_maxone(x, decoder, beam_size):

	p_y = decoder.predict(x, verbose=0)
	decoded_sentence = get_predict_list(p_y)[0]
		
	return decoded_sentence

def search_path_beam(x, decoder, beam_size):
	p_y = decoder.predict(x, verbose=0)
	tf.nn.ctc_beam_search_decoder(p_y, p_y.shape[0], beam_size, 1, False)

def decode_sequence_teach(encoder, decoder, X, beam_size=2):
	
	#reshape X to one hot matrix
	x = np.zeros((1, X.shape[0], cfg.input_vocab_size))
	for t in range(X.shape[0]):
		x[0, t, X[t]] = 1.0
	
	# Encode the input as state vectors.
	states_value = encoder.predict(x)
# 	print states_value[0].shape
# 	print states_value[1].shape
	decoded_sentence =  search_path_beam1_teach(decoder=decoder, init_state=states_value, beam_size=beam_size)
	
	return " ".join(decoded_sentence)



def search_path_maxone_teach(decoder, init_state, beam_size):
	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1, 1, cfg.output_vocab_size))
	# Populate the first character of target sequence with the start character.
	target_seq[0, 0, cfg.dic_output_word2i['<GO>']] = 1.
	states_value = init_state
	# Sampling loop for a batch of sequences
	# (to simplify, here we assume a batch of size 1).
	stop_condition = False
	decoded_sentence = []
	while not stop_condition:
		output_tokens_prob, h, c = decoder.predict([target_seq] + states_value)
		
		# Sample a token
		sampled_token_index = np.argmax(output_tokens_prob[0, -1, :])
		# beam search
		arr = output_tokens_prob[0, -1, :]
		result = heapq.nlargest(beam_size, enumerate(arr),itemgetter(1))
		#extract the n maxmium indies
		max_index = map(lambda x:x[0], result)
		
		sampled_token = cfg.dic_output_i2word[sampled_token_index]
		
		if sampled_token != cfg.end_flg and sampled_token != cfg.start_flg:
			decoded_sentence.append(sampled_token)
		
		# Exit condition: either hit max length
		# or find stop character.
		if (sampled_token == cfg.end_flg or sampled_token == cfg.pad_flg or len(decoded_sentence) > cfg.max_output_len):
			stop_condition = True
		
		# Update the target sequence (of length 1).
		target_seq = np.zeros((1, 1, cfg.output_vocab_size))
		target_seq[0, 0, sampled_token_index] = 1.
		
		# Update states
		states_value = [h, c]
		
	return decoded_sentence

def search_path_viterbi_teach(decoder, init_state, beam_size):
	def get_target_seq(state):
		target_seq = np.zeros((1, 1, cfg.output_vocab_size))
		target_seq[0, 0, state] = 1.
		return target_seq
	
	def get_state_list(time):
		states = []
		if time < 1:
			states = [cfg.dic_output_word2i[cfg.start_flg]]
		else:
			states = range(cfg.output_vocab_size)
		return states
	# Generate empty target sequence of length 1.
	# Populate the first character of target sequence with the start character.
	#     target_seq = get_target_seq(cfg.dic_output_word2i[cfg.start_flg])
	
	#forward state
	decoded_sentence = []
	states_value = init_state
	best_score = np.zeros((cfg.max_output_len_decode, cfg.output_vocab_size))
	best_tokens = np.zeros((cfg.max_output_len_decode, cfg.output_vocab_size))
	best_tokens[0] = cfg.dic_output_word2i[cfg.start_flg]
	
	pre_states_value_dic = {cfg.dic_output_word2i[cfg.start_flg]:init_state}
	for t in range(1, cfg.max_output_len_decode):
		log_prob_list = []
		temp_dic = {}
		state_pre_list = get_state_list(t - 1)
		for state_pre in state_pre_list:
			target_seq = get_target_seq(state_pre)
			states_value = pre_states_value_dic[state_pre]
			output_tokens_prob, h, c = decoder.predict([target_seq] + states_value)
			log_prob = -np.log10(output_tokens_prob)
			log_prob = np.squeeze(log_prob)
			log_prob += best_score[t - 1, state_pre]
			log_prob_list.append(log_prob)
			temp_dic[state_pre] = [h, c]
		
		log_prob_arr = np.vstack(log_prob_list)
		max_token_indies = np.argmin(log_prob_arr, axis=0)
		if t == 1:
			best_tokens[t] = state_pre_list[0]
		else:
			best_tokens[t] = max_token_indies
			
		best_score[t] = log_prob_arr[max_token_indies, range(log_prob_arr.shape[1])]
		pre_states_value_dic = {}
		for i,j in enumerate(max_token_indies):
			if t == 1:
				pre_states_value_dic[i] = temp_dic[cfg.dic_output_word2i[cfg.start_flg]]
			else:
				pre_states_value_dic[i] = temp_dic[j]

	#backward stage
	pre_index = np.argmin(best_score[-1])
	sampled_token = cfg.dic_output_i2word[pre_index]
	if sampled_token != cfg.end_flg and sampled_token != cfg.start_flg and sampled_token != cfg.pad_flg:
		decoded_sentence.append(sampled_token)
	for t in range(1, cfg.max_output_len_decode - 1)[::-1]:
		pre_index = int(best_tokens[t, pre_index])
		sampled_token = cfg.dic_output_i2word[pre_index]
		if sampled_token != cfg.end_flg and sampled_token != cfg.start_flg and sampled_token != cfg.pad_flg:
			decoded_sentence.append(sampled_token)

	decoded_sentence.reverse()

#     while True:
#         output_tokens_prob, h, c = decoder.predict([target_seq] + states_value)
#         
#         # Sample a token
#         sampled_token_index = np.argmax(output_tokens_prob[0, -1, :])
#         # beam search
#         arr = output_tokens_prob[0, -1, :]
#         result = heapq.nlargest(beam_size, enumerate(arr),itemgetter(1))
#         #extract the n maxmium indies
#         max_index = map(lambda x:x[0], result)
#         
#         sampled_token = cfg.dic_output_i2word[sampled_token_index]
#         
#         if sampled_token != cfg.end_flg and sampled_token != cfg.start_flg:
#             decoded_sentence.append(sampled_token)
#         
#         # Exit condition: either hit max length
#         # or find stop character.
#         if (sampled_token == cfg.end_flg or sampled_token == cfg.pad_flg or len(decoded_sentence) > cfg.max_output_len):
#             stop_condition = True
#         
#         # Update the target sequence (of length 1).
#         target_seq = np.zeros((1, 1, cfg.output_vocab_size))
#         target_seq[0, 0, sampled_token_index] = 1.
#         
#         # Update states
#         states_value = [h, c]
	
	return decoded_sentence


def search_path_beam1_teach(decoder, init_state, beam_size):
	def get_target_seq(state):
		target_seq = np.zeros((1, 1, cfg.output_vocab_size))
		target_seq[0, 0, state] = 1.
		return target_seq
	
	def initial_state_function(x):
# 		init_state = encoder.predict(x)
		return init_state
	
	def generate_function(X, Y_tm1, state_tm1):
		states = []
		probs = []
		extras = []
		for y, s in zip(Y_tm1, state_tm1):
			x = get_target_seq(y)
			output_tokens_prob, h, c = decoder.predict([x, s[0], s[1]])
			states.append([h,c])
			probs.append(np.squeeze(output_tokens_prob))
			extras.append(c)
		return states, np.vstack(probs), np.vstack(extras)
		
	hypotheses = beam_search(initial_state_function, generate_function, None, 
			cfg.dic_output_word2i[cfg.start_flg], cfg.dic_output_word2i[cfg.end_flg], 
			beam_width=beam_size, num_hypotheses=1, max_length = cfg.max_output_len_decode)
	
	generated_tokens = []
	for hypothesis in hypotheses:
		generated_indices = hypothesis.to_sequence_of_values(set([cfg.dic_output_word2i[cfg.start_flg], cfg.dic_output_word2i[cfg.end_flg]]))
		generated_tokens = [cfg.dic_output_i2word[i] for i in generated_indices]
# 		print(" ".join(generated_tokens))
	return generated_tokens

def search_path_beam_teach(decoder, init_state, beam_size):
	def get_target_seq(state):
		target_seq = np.zeros((1, 1, cfg.output_vocab_size))
		target_seq[0, 0, state] = 1.
		return target_seq
	
	def get_state_list(time):
		states = []
		if time < 1:
			states = [cfg.dic_output_word2i[cfg.start_flg]]
		else:
			states = best_tokens[time]
		return states
	# Generate empty target sequence of length 1.
	# Populate the first character of target sequence with the start character.
#     target_seq = get_target_seq(cfg.dic_output_word2i[cfg.start_flg])

	#forward state
	decoded_sentence = []
	states_value = init_state
	best_score = np.zeros((cfg.max_output_len_decode, beam_size))
	best_tokens = np.zeros((cfg.max_output_len_decode, beam_size))
	best_tokens[0] = cfg.dic_output_word2i[cfg.start_flg]
	
	pre_states_value_dic = {cfg.dic_output_word2i[cfg.start_flg]:init_state}
	for t in range(1, cfg.max_output_len_decode):
		log_prob_list = []
		temp_dic = {}
		for state_pre in get_state_list(t - 1):
			target_seq = get_target_seq(state_pre)
			states_value = pre_states_value_dic[state_pre]
			output_tokens_prob, h, c = decoder.predict([target_seq] + states_value)
			log_prob = -np.log10(output_tokens_prob)
			log_prob = np.squeeze(log_prob)
			log_prob += best_score[t - 1, state_pre]
			log_prob_list.append(log_prob)
			temp_dic[state_pre] = [h, c]
		
		log_prob_arr = np.vstack(log_prob_list)
		max_token_indies = np.argmin(log_prob_arr, axis=0)
		result = heapq.nlargest(beam_size, enumerate(max_token_indies), itemgetter(1))
		max_token_indies = map(lambda x:x[0], result)
		
		best_tokens[t] = max_token_indies
		best_score[t] = log_prob_arr[max_token_indies, range(log_prob_arr.shape[1])]
		
		pre_states_value_dic = {}
		for i,j in enumerate(max_token_indies):
			pre_states_value_dic[i] = temp_dic[j]

	#backward stage
	pre_index = np.argmin(best_score[-1])
	sampled_token = cfg.dic_output_i2word[pre_index]
	if sampled_token != cfg.end_flg and sampled_token != cfg.start_flg and sampled_token != cfg.pad_flg:
		decoded_sentence.append(sampled_token)
	for t in range(1, cfg.max_output_len_decode)[::-1]:
		pre_index = best_tokens[t, pre_index]
		sampled_token = cfg.dic_output_i2word[pre_index]
		if sampled_token != cfg.end_flg and sampled_token != cfg.start_flg and sampled_token != cfg.pad_flg:
			decoded_sentence.append(sampled_token)

	decoded_sentence.reverse()
	

	return decoded_sentence    
def run_evalute_and_split():
	df_test = pd.read_csv('../data/train_cls0.csv')
	df_test['p_new_class'] = -1
# 	test_feat = np.load('../data/en_train_classify.npz')
	y_t_cls = data_process.load_numpy_data('../data/train_y_2class.npy')
	
	test_feat = np.load("../data/train_cls0.npz")
	
	x_t_cls = test_feat['x_t_c']
	x_t = test_feat['x_t']
	y_t = test_feat['y_t']
	
	x_t_e = np.load('../data/extend_features.npy')
	
# 	x_t_cls = [x_t_cls, x_t_e]
	
	mod_classifier = gen_model_class_2()
	print(mod_classifier.summary())
	mod_classifier.load_weights("../model/class_cnn_c2_weights.98-0.0005-1.0000-0.0004-1.0000.hdf5")
	_, x_list, cls_df_list, cls_y_list, y_list = split_by_classifier(mod_classifier, 2, x_t_cls, 
																	np.hstack([x_t, x_t_cls]), 
																	df_test, y_t_cls, y_t)
	

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

def eval_trained_model(batch_size=256):
	path = "../model/tf_teach_att_bl3_bl1_c0.32-0.00588-0.99792-0.99115-98.25900.ckpt-83808"
	with tf.Session() as sess:
		model = restore_tf_model(path, sess, batch_size, False)

		X_train = np.loadtxt("../data/X_train.txt")
		Y_train = np.loadtxt("../data/Y_train.txt")
		X_valid = np.loadtxt("../data/X_valid.txt")
		Y_valid = np.loadtxt("../data/Y_valid.txt")
		dataset = NumpySeqData(cfg.pad_flg_index, cfg.end_flg_index)
		dataset.load(X_train, Y_train, X_valid, Y_valid, cfg.vocab_i2word)
		dataset.build()

		print '================== start evaluation of trained model =================='
		right, wrong = 0.0, 0.0
		#total_batch = X_valid.shape[0]
		#for step, data_dict in enumerate(dataset.val_datas(batch_size, False)):
		total_batch = math.ceil(X_train.shape[0] / float(batch_size))
		for step, data_dict in enumerate(dataset.train_datas(batch_size, False)):
			print '---> Finished batch : %d/%d' % (step, total_batch)
			feed_dict = model.make_feed_dict(data_dict)
			beam_result_ids = sess.run(model.beam_search_result_ids, feed_dict)
			beam_result_ids = beam_result_ids[:, :, 0]
			print beam_result_ids.shape
			if step == 0:
				print(data_dict['decoder_inputs'][:5])
				print(beam_result_ids[:5])
			now_right, now_wrong, infos = dataset.eval_result(data_dict['encoder_inputs'], data_dict['decoder_inputs'], beam_result_ids, step, batch_size)
			right += now_right
			wrong += now_wrong
		print "Right: {}, Wrong: {}, Accuracy: {:.2}%".format(right, wrong, 100*right/float(right+wrong))



if __name__ == "__main__":
# 	data_process.extract_val_ret_err()

# 	data_process.gen_train_feature_from_files('../data/ext2/')
# 	data_process.gen_filter_duplicated_data_from_files('../data/ext2/', is_filter_by_xy=True)
# 	data_process.down_sampling_from_files('../data/ext2/')
# 	df = pd.read_csv('../data/ext/output-00001-of-00100_train_cls0_filtered.csv')
# 	data_process.filter_reduplicated_xy_data(df)
# 	data_process.gen_constant_dict()
# 	data_process.gen_alpha_table()
# 	data_process.gen_out_vocab()
# 	run_evalute()
#  	df = pd.read_csv('../data/ext/output-00003-of-00100_train_cls0_filtered.csv')
# 	data_process.display_token_info(df, "NU.nl", "before")
# 	data_process.gen_constant_dict()
# 	data_process.add_class_info('../data/en_train_filted_all.csv', "../data/en_train_filted_class.csv")
# 	data_process.add_class_info('../data/en_test_2.csv', "../data/en_test_class.csv")
# 	df = pd.read_csv('../data/en_test_class.csv')
# 	
# 	
# # 	run_evalute_and_split()
# 	data_process.gen_train_feature(df)
# 	df = pd.read_csv('../data/en_test_class.csv')
# # # # # 	
# 	data_process.gen_test_feature(df)
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
# 	experiment_teaching1(cls_id=1, pre_train_model_file = "../model/teach_l1_l1_c1_weights.134-0.0023-0.9994-0.0021-0.9994.hdf5")
#  	del df['len']
#  	print 'saved cnt info.'
#  	data_process.gen_one2one_dict(df)
# 	df = pd.read_csv('../data/en_test_class.csv')
# 	data_process.gen_super_long_items(df, cfg.max_input_len - 2, '../data/en_test_long_tokens.csv')
# 	
# 	df = pd.read_csv('../data/en_train_filted_class.csv')
# 	data_process.gen_super_long_items(df, cfg.max_input_len - 2, '../data/en_train_long_tokens.csv')
	
# 	experiment_simple_gru(nb_epoch=100, input_num=10000, cls_id=0, pre_train_model_file=None)
# 	experiment_teaching0()
# 	evalute_acc('../data/test_ret.csv', '../data/test_ret_err.csv')
# 	export_feature_data()
# 	run_evalute()
# 	experiment_simple_lstm()
# 	df = pd.read_csv('../data/en_test_2.csv')
# 	token = df.at[13380, "before"]
# 	print cfg.i2w(683)
# 	print cfg.i2w(1495)
# 	print cfg.i2w(1307)
# 	print cfg.i2w(2984)
# 	print cfg.i2w(256)
# 	print cfg.i2w(530)
# 	print cfg.i2w(1755)
# 	print cfg.i2w(2401)
# 	print cfg.i2w(1883)
# 	print cfg.i2w(530)
# 	print cfg.i2w(1307)
# 	print cfg.i2w(1464)
# 	print cfg.i2w(1952)
	
# 	print cfg.i2w(1307)
# 	print cfg.i2w(256)
# 	print cfg.i2w(1136)
# 	print cfg.i2w(1307)
# 	print cfg.i2w(1056)
# 	print cfg.i2w(256)
# 	print cfg.i2w(1307)
# 	print cfg.i2w(1464)
# 	print cfg.i2w(256)
# 	print cfg.i2w(3903)
# 	print cfg.i2w(1755)
# 	print cfg.i2w(2401)
# 	print cfg.i2w(1883)
# 	print cfg.i2w(3903)
# 	print cfg.i2w(1307)
# 	print cfg.i2w(1464)
# 	print cfg.i2w(1952)
	
	
# 	print cfg.i2w(406)
# 	print cfg.i2w(2044)
# 	print cfg.i2w(2044)
# 	print cfg.i2w(1883)

# 	out = data_process.recover_y_info([1, 1307, 256, 1136, 1307, 1056, 1307, 1464, 256, 3903, 1755, 2401, 1883, 3903, 1307, 1464, 1952, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
# 	print out 
# 	run_normalize(False, 1000, False, cfg.data_args_test)
# 	experiment_teaching_tf(batch_size=256, nb_epoch=100, input_num=0, cls_id=0,
# 						   file_head="tf_teach_att384_bl4_bl1_c", pre_train_model_file=None)

	experiment_teaching_tf(batch_size=256, nb_epoch=100, input_num=0, test_size=100000, cls_id=0,
						   file_head="tf_teach_sche_att_bl4_bl1_c", is_debug=False, 
# 						   pre_train_model_prefix="../model/tf/tf_teach_sche_att_bl4_bl1_c0.00-0.17655-0.95019-0.87283-77.58200.ckpt-9928",
						   pre_train_model_prefix=None)

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
# 	eval_trained_model(batch_size=256)
