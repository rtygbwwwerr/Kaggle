import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from collections import Counter
import seaborn as sns
from sklearn import preprocessing
import os
import re
from glob import glob
from pandas.tseries.offsets import _is_normalized
sys.path.append('../../')
from base import data_util
from config import Config as cfg
from scipy.io import wavfile
from python_speech_features import mfcc
from python_speech_features import logfbank


def standardization(X):
	x_t = preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=True)
	return x_t
def call_feature_func(data, feats_name, is_normalized=False):
	feats = pd.DataFrame()
	for name in feats_name:
		func_name = "extract_{feat}_feature".format(feat = name)
		feats[name] = eval(func_name)(data, is_normalized)
	return feats





def extract_id_feature(data, is_normalized=False):
	def q_val(val):
		return val
	vals = data.apply(lambda x: q_val(x['id']), axis=1, raw=True)
	if is_normalized:
		vals = standardization(vals)
	return vals




def extract_all_features(df_list, is_normalized=False):
	data_all = pd.concat(df_list)
	feats_name = [
				'id',
				]

	feats = call_feature_func(data_all, feats_name, is_normalized)
	return feats








def extract_y_info(data):
	print 'Need to be implemented'
	return None





def gen_data(df_train, df_test, is_resample=False, is_normalized=False):
	x_train = extract_all_features([df_train], is_normalized)
	y_train = extract_y_info(df_train)



	x_test = extract_all_features([df_test], is_normalized)
	y_test = extract_y_info(df_test)



	x_t = x_train
	y_t = y_train
	if is_resample:
		x_t, y_t = data_util.resample(x_train, y_train)
	return x_t, y_t, x_test, y_test

def load_data(data_dir="../data/"):
	""" Return 2 lists of tuples:
	[(class_id, user_id, path), ...] for train
	[(class_id, user_id, path), ...] for validation
	"""
	# Just a simple regexp for paths with three groups:
	# prefix, label, user_id
	pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
	all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
	
	with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
		validation_files = fin.readlines()
	valset = set()
	for entry in validation_files:
		r = re.match(pattern, entry)
		if r:
			valset.add(r.group(3))

	possible = set(cfg.POSSIBLE_LABELS)
	train, val = [], []
	for entry in all_files:
		r = re.match(pattern, entry)
		if r:
			label, uid = r.group(2), r.group(3)
			if label == '_background_noise_':
				label = 'silence'
			if label not in possible:
				label = 'unknown'
			
			label_id = cfg.n2i(label)
			
			sample = (label_id, uid, entry)
			if uid in valset:
				val.append(sample)
			else:
				train.append(sample)
	
	print('There are {} train and {} val samples'.format(len(train), len(val)))
	return train, val


def get_wav(data, resampline_sil_rate=20, is_normalization=True):
	x = []
	y = []
	for (label_id, uid, fname) in data:
# 		print fname
		_, wav = wavfile.read(fname)
# 		wavfile.write("../data/test_original.wav", cfg.sampling_rate, wav)
		wav = wav.astype(np.float32) / np.iinfo(np.int16).max
		
		# be aware, some files are shorter than 1 sec!
		if len(wav) < cfg.sampling_rate:
			continue
		if is_normalization:
			wav = standardization(wav)
		
# 		print feat.shape

# 		# we want to compute spectograms by means of short time fourier transform:
# 		specgram = signal.stft(
# 			wav,
# 			400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
# 			160,  # 16000 * 0.010 -- default stride
# 		)
		
		

		#sampling rate means 1sec long data
		L = cfg.sampling_rate
		# let's generate more silence!
		samples_per_file = 1 if label_id != cfg.n2i(cfg.sil_flg) else resampline_sil_rate
		for _ in range(samples_per_file):
			if len(wav) > L:
				beg = np.random.randint(0, len(wav) - L)
# 				print len(wav)
			else:
				beg = 0
			wav = wav[beg: beg + L]
			x.append(wav)
			y.append(np.int32(label_id))
	return x, y


def get_features(data, extract_func=mfcc):
	
	
	output = map(lambda x : extract_func(x, cfg.sampling_rate), data)
	
	return output
def pad_data(data):
	lengths = map(lambda x : len(x), data)
	L = max(lengths)
	npz_data = np.zeros((len(data), L, data[0].shape[1]))
	for i, x in enumerate(data):
		pad_x = x
		for j in range(len(x), L):
			pad_x = np.row_stack(x, np.ones(x.shape[1]) * cfg.non_flg_index)
		npz_data[i] = pad_x
	return npz_data
		
def gen_train_feature(data_dir="../data/"):
	train, val = load_data(data_dir)
	x, y = get_wav(train)
	x = get_features(x)
	x = pad_data(x)
	
	v_x, v_y = get_wav(val)
	v_x = get_features(v_x)
	v_x = pad_data(v_x)
	
	np.savez("../data/train/train.npz", x = x, y = y)
	np.savez("../data/valid/valid.npz", x = v_x, y = v_y)
	print "completed gen training data..."
	
def test():
	gen_train_feature()
	




if __name__ == "__main__":
	test()
