import math
import numpy as np
import pandas as pd
from collections import Counter
from imblearn import over_sampling as osamp
import inspect
import sys
import os
from macpath import split
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

def isDigitType(value):
	is_flg = isinstance(value, int) or isinstance(value, float) or \
			isinstance(value, np.float) or isinstance(value, np.int) or \
			isinstance(value, np.float32) or isinstance(value, np.int32) or \
			isinstance(value, np.float64) or isinstance(value, np.int64) or \
			isinstance(value, np.float128) or isinstance(value, np.int16) or \
			isinstance(value, np.float16) or isinstance(value, np.int8)
	return is_flg

def isDigitArrType(value):
	type = value.dtype
	is_flg = (type == int) or (type == float) or \
			(type == np.float) or (type == np.int) or \
			(type ==  np.float32) or (type ==  np.int32) or \
			(type == np.float64) or (type == np.int64) or \
			(type == np.float128) or (type == np.int16) or \
			(type == np.float16) or (type == np.int8)
			
	return is_flg

def floatrange(start,stop,steps):
	''' Computes a range of floating value.
	   
		Input:
			start (float)  : Start value.
			end   (float)  : End value
			steps (integer): Number of values
	   
		Output:
			A list of floats
	   
		Example:
			>>> print floatrange(0.25, 1.3, 5)
			[0.25, 0.51249999999999996, 0.77500000000000002, 1.0375000000000001, 1.3]
	'''
	return [start+float(i)*(stop-start)/(float(steps)-1) for i in range(steps)]

def resample(x_df, y):
	#  	ros = osamp.RandomOverSampler(random_state=0)
	#  	x_t, y_t = ros.fit_sample(x_df, y)
	# 	x_t, y_t = SMOTEENN(random_state=42).fit_sample(x_df, y)
	# 	x_t, y_t = SMOTETomek(random_state=42).fit_sample(x_df, y)
	x_t, y_t = osamp.SMOTE(random_state=0, kind='svm').fit_sample(x_df, y)
	#	x_t, y_t = osamp.ADASYN(random_state=0, n_neighbors=5).fit_sample(x_df, y)
	
	# 	x_t = preprocessing.scale(x_t, axis=0, with_mean=True, with_std=True, copy=True)
	
	
	
	head_name = x_df.columns.values.tolist()
	x_t = pd.DataFrame(x_t,  columns=head_name)
	
	return x_t, y_t 

def info_gain(datalist, column_name, label_name):
	data_all = []
	for dataset in datalist:
		data_all.extend(dataset)
			
	freqs_dict = data_all[column_name].value_counts().to_dict()
	N = float(len(data_all))

	Hct = 0
	Npo = float(len(data_all[data_all[label_name] > 0]))
	Nno = float(len(data_all[data_all[label_name] == 0]))
	Hc = -(Npo/N)* math.log(Npo/N) - (Nno/N) * math.log(Nno/N) 
	
	for k, v in freqs_dict.iteritems():
		pt = v / N
# 		H = H - p * math.log(p)
			
		Np = float(len(data_all[(data_all[column_name]==k) & (data_all[label_name] > 0)]))
		Nn = float(len(data_all[(data_all[column_name]==k) & (data_all[label_name] == 0)]))
		
		Pct1 = Np / v
		Pct0 = Nn / v
		
		Hct = Hct - pt * (Pct0 * math.log(Pct0) - Pct1 * math.log(Pct1))
		
	IGN = Hc - Hct
	print 'IGN of %s:%f'%(column_name, IGN)
	return IGN

# def call_feature_func(data, is_normalized, *features):
# 	frame = sys._getframe(1)
# 	code = frame.f_code
# 
# # 	print "frame depth = ", 1
# # 	print "func name = ", code.co_name
# # 	print "func filename = ", code.co_filename
# # 	print "func lineno = ", code.co_firstlineno
# # 	print "func locals = ", frame.f_locals
# 	
# 	root = code.co_filename.split('/')[-3]
# 	src = code.co_filename.split('/')[-2]
# 	module = os.path.basename(code.co_filename).split('.')[0]
# 	module = '{root_name}.{src_name}.{module_name}'.format(root_name=root, src_name=src, module_name=module)
# # 	eval("from {root_name}.{src_name} import {name}".format(name = module_name, root_name = root, src_name = src))
# 	sys.path.append('../')
# 	__import__(module)
# 	
# 	feat = pd.DataFrame()
# 	for name in features:
# 		func_name = "{module_name}.extract_{feat}_feature".format(module_name = module, feat = name)
# 		feat[name] = eval(func_name)(data, is_normalized)
# 	
# 	return feat
def q_val_delta_date(start, end, start_format, end_format, default=None):
	try:
		date1 = datetime.strptime(start, start_format)
		date2 = datetime.strptime(end, end_format)
		delta = date1 - date2
		return delta.total_seconds()
	except (TypeError, ValueError), e:
		print e.message
		return default

def draw_correlation_matrix(df_feats):
	sns.heatmap(df_feats.corr(method='pearson', min_periods=1), cmap="BrBG", annot=True)
	plt.show()
	
def detect_intersection_ids(df_train, df_test):
	test_ids = set(df_test.user_id.unique())
	train_ids = set(df_train.user_id.unique())
	intersection_count = len(test_ids & train_ids)
	
	print "train id num:%d, test id num:%d, intersection id num:%d."%(len(train_ids), len(test_ids), intersection_count)
	
	return intersection_count == len(test_ids)

def get_arrays_lengths(a, PAD_ID):
	lens = 0
	if PAD_ID is None:
		lens = get_sequence_lengths(a)
	else:
		lens = np.sum(a != PAD_ID, axis=1)
	return lens
	

def get_sequence_lengths(sequences):
# 	for s in sequences:
# 		s.index(end_id)
	lengths = np.asarray([len(s) for s in sequences], dtype=np.int32)
	return lengths

def pad_sequences(sequences, maxlen=None, dtype=np.float32,
				  padding='post', truncating='post', value=0.):
	'''Pads each sequence to the same length: the length of the longest
	sequence.
		If maxlen is provided, any sequence longer than maxlen is truncated to
		maxlen. Truncation happens off either the beginning or the end
		(default) of the sequence. Supports post-padding (default) and
		pre-padding.

		Args:
			sequences: list of lists where each element is a sequence
			maxlen: int, maximum length
			dtype: type to cast the resulting sequence.
			padding: 'pre' or 'post', pad either before or after each sequence.
			truncating: 'pre' or 'post', remove values from sequences larger
			than maxlen either in the beginning or in the end of the sequence
			value: float, value to pad the sequences to the desired value.
		Returns
			x: numpy array with dimensions (number_of_sequences, maxlen)
			lengths: numpy array with the original sequence lengths
	'''
	lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

	nb_samples = len(sequences)
	if maxlen is None:
		maxlen = np.max(lengths)

	# take the sample shape from the first non empty sequence
	# checking for consistency in the main loop below.
	sample_shape = tuple()
	for s in sequences:
		if len(s) > 0:
			sample_shape = np.asarray(s).shape[1:]
			break

	x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
	for idx, s in enumerate(sequences):
		if len(s) == 0:
			continue  # empty list was found
		if truncating == 'pre':
			trunc = s[-maxlen:]
		elif truncating == 'post':
			trunc = s[:maxlen]
		else:
			raise ValueError('Truncating type "%s" not understood' % truncating)

		# check `trunc` has expected shape
		trunc = np.asarray(trunc, dtype=dtype)
		if trunc.shape[1:] != sample_shape:
			raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
							 (trunc.shape[1:], idx, sample_shape))

		if padding == 'post':
			x[idx, :len(trunc)] = trunc
		elif padding == 'pre':
			x[idx, -len(trunc):] = trunc
		else:
			raise ValueError('Padding type "%s" not understood' % padding)
	return x, lengths
   
def sparse_tuple_from(sequences, dtype=np.int32):
	"""Create a sparse representention of x.
	Args:
		sequences: a list of lists of type dtype where each element is a sequence
	Returns:
		A tuple with (indices, values, shape)
	"""
	indices = []
	values = []
	
	for n, seq in enumerate(sequences):
		indices.extend(zip([n]*len(seq), range(len(seq))))
		values.extend(seq)
	
	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
	
	return indices, values, shape

def drop_outliers(dataframe, n, features):
	"""
	Takes a dataframe dataframe of features and returns a list of the indices
	corresponding to the observations containing more than n outliers according
	to the Tukey method (Tukey JW., 1977).
	"""
	outlier_indices = []
	
	# iterate over features(columns)
	for col in features:
		# 1st quartile (25%)
		Q1 = np.percentile(dataframe[col], 25)
		# 3rd quartile (75%)
		Q3 = np.percentile(dataframe[col], 75)
		# Interquartile range (IQR)
		IQR = Q3 - Q1
		
		# outlier step
		outlier_step = 1.5 * IQR
		
		# Determine a list of indices of outliers for feature col
		outlier_list_col = dataframe[(dataframe[col] < Q1 - outlier_step) | (dataframe[col] > Q3 + outlier_step )].index
		
		# append the found outlier indices for col to the list of outlier indices 
		outlier_indices.extend(outlier_list_col)
	
	# select observations containing more than 2 outliers
	outlier_indices = Counter(outlier_indices)		
	outliers = list( k for k, v in outlier_indices.items() if v > n )
	
	print 'drop outlier rows:%s'%str(outliers)
	df_train = dataframe.drop(outliers, axis = 0).reset_index(drop=True)
	
	return df_train
