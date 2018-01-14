import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from collections import Counter
from keras.utils.np_utils import to_categorical
import seaborn as sns
from sklearn import preprocessing
import os
import re
from glob import glob
import librosa
import random
from vad import VoiceActivityDetector
from sklearn.decomposition import PCA
import shutil
from shutil import copyfile
sys.path.append('../../')
from base import data_util
from config import Config as cfg
from scipy.fftpack import fft
from scipy.io import wavfile
import wave
from scipy import signal
from python_speech_features import mfcc, logfbank, fbank, delta, ssc
from keras.preprocessing import sequence
import audioop
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from vad import VoiceActivityDetector
from base.data_util import gen_input_paths
import math
from library import audioFeatureExtraction
# import pylab as pl
import datetime, time
cfg.init()

VAD = VoiceActivityDetector()

def make_if_not_exist(dir):
	if not os.path.exists(dir):
		print "created dir:" + dir
		os.mkdir(dir)
		
def pad_audio(samples, max_len=cfg.sampling_rate):
	if len(samples) >= max_len: 
		return samples
	else: 
		N = max_len - len(samples)
		pad_left = N / 2
		pad_right = N - pad_left
		return np.pad(samples, pad_width=(pad_left, pad_right), mode='constant', constant_values=(0, 0))

def read_wav(fname, is_divide_max=True):
	sr, wav = wavfile.read(fname)
	wav = pad_audio(wav)
	if is_divide_max:
		wav = wav.astype(np.float32) / np.iinfo(np.int16).max
	return sr, wav

def init_fake_noise(name=None, path='../data/train/ext0/'):
	noises = []
	if name == None:
		name = ""
	paths = gen_input_paths(path, file_beg_name=name, file_ext_name='.wav')	
	
	for path in paths:
		_, wav = read_wav(path, True)
		noises.append(wav)
	return noises

def init_noise_array(paths):
	
	data = []
	for path in paths:
		wpaths = gen_input_paths(path, file_ext_name=".wav")
		for fname in wpaths:
			if '-' not in fname:
				_, wav = wavfile.read(fname)
				wav = wav.astype(np.float32) / np.iinfo(np.int16).max
				
				# be aware, some files are shorter than 1 sec!
				if len(wav) < cfg.sampling_rate:
					continue
				N = len(wav) / cfg.sampling_rate
				for i in range(N):
					data.append(wav[i*cfg.sampling_rate:(i+1)*cfg.sampling_rate])
	print "build up {} items' noise set.".format(len(data))
	return data

noise_array = init_noise_array(['../data/train/audio/_background_noise_'])
noise_dict = {}
noise_dict['doing_the_dishes'] = init_fake_noise('doing_the_dishes')
noise_dict['dude_miaowing'] = init_fake_noise('dude_miaowing')
noise_dict['exercise_bike'] = init_fake_noise('exercise_bike')
noise_dict['pink_noise'] = init_fake_noise('pink_noise')
noise_dict['running_tap'] = init_fake_noise('running_tap')
noise_dict['white_noise'] = init_fake_noise('white_noise')
noise_dict['real'] = init_fake_noise(name=None, path='../data/train/ext1/')

def standardization(X):
# 	x_t = preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=True)
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	std = std + 1e-12
	x_t = (X - mean) / std
	
	
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





# def data_prepare(input_num, file_head, file_type, data_names=['x','x_wav', "y_c"]):
# 	train_data = get_data_from_files("../data/train/", data_names, file_type, input_num)	
# 	valid_data = get_data_from_files("../data/valid/", data_names, file_type, input_num)
# 	
# 	
# 	file_head = "{}_{}".format(file_head, file_type)
# 	
# 	print "train feature shape:{}*{}".format(x_train[0].shape[0], x_train[0].shape[1])
# 	print "train items num:{0}, valid items num:{1}".format(x_train.shape[0], x_valid.shape[0])
# 	
# 	return train_data, valid_data

def custom_fft(y, fs):
	T = 1.0 / fs
	N = y.shape[0]
	yf = fft(y)
	xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
	# FFT is simmetrical, so we take just the first half
	# FFT is also complex, to we take just the real part (abs)
	vals = 2.0/N * np.abs(yf[0:N//2])
	return xf, vals


	
def logspecgram(x, sample_rate, window_size=20,
					  step_size=10, eps=1e-10):
	nperseg = int(round(window_size * sample_rate / 1e3))
	noverlap = int(round(step_size * sample_rate / 1e3))
	freqs, times, spec = signal.spectrogram(x,
											  fs=sample_rate,
											  window='hann',
											  nperseg=nperseg,
											  noverlap=noverlap,
											  detrend=False)
	return np.log(spec.T.astype(np.float32) + eps)

def load_data(data_dir="../data/", outlier_path=None, include_labels=None):
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

# 	possible = set(cfg.POSSIBLE_LABELS)
	train, val = [], []
	for entry in all_files:
		r = re.match(pattern, entry)
		if r:
			label, uid = r.group(2), r.group(3)
			if label == '_background_noise_':
# 				label = 'silence'
			#skip background files, because we will handle them in ext0
				continue
# 			if label not in possible:
# 				label = 'unknown'
			if include_labels is not None and len(include_labels) > 0 and label not in include_labels:
				continue
			sample = (label, entry)
			if uid in valset:
				val.append(sample)
			else:
				train.append(sample)
	
	if outlier_path is not None:
		outlier_set = get_outlier_set(outlier_path)
		train = filter_outlier_files(outlier_set, train)
		val = filter_outlier_files(outlier_set, val)
		print "we have deleted {} bad records".format(len(outlier_set))
	print('There are {} train and {} val samples'.format(len(train), len(val)))
	return train, val


# 	return np.pad(samples, pad_width=(max_len - len(samples), 0), mode='constant', constant_values=(0, 0))

def gen_silence_data(resampline_sil_rate=200, output="../data/train/ext0/"):
	inputs = gen_input_paths("../data/train/audio/_background_noise_/", file_ext_name=".wav")
	L = cfg.sampling_rate
	for fname in inputs:
		rate, wav = wavfile.read(fname)
		name = os.path.basename(fname)[:-4]

		for id in range(resampline_sil_rate):
			
			if len(wav) > L:
				beg = np.random.randint(0, len(wav) - L)
# 				print len(wav)
			else:
				beg = 0
			wav = wav[beg: beg + L]
			filename = "{}{}_{}.wav".format(output, name, id)
			wavfile.write(filename, rate, wav)
			


def read_wav_and_labels(data, is_normalization=True):
	x = []
	labels = []
	fnames = []
# 	y_c = []
# 	y_w = []
	i = 0
	for (label, fname) in data:
# 		print fname
		_, wav = read_wav(fname)
		
# 		# be aware, some files are shorter than 1 sec!
# 		if len(wav) < cfg.sampling_rate:
# 			continue
# 		if is_normalization:
# 			wav = standardization(wav)
		fnames.append(fname)
		labels.append(label)
		x.append(wav)
		i += 1
		if i % 10000 == 0:
			print "read {} files".format(i)
	print "completed read {} wav items".format(len(x))
# 	y = to_categorical(y, num_classes = cfg.CLS_NUM)
	return x, labels, fnames

def warp_sil_labels(label):
# 	labels = [cfg.start_flg, cfg.sil_flg]
	labels = [cfg.sil_flg]
	if type(label) is list:
		labels.extend(label)
	else:
		labels.append(label)
	labels.append(cfg.sil_flg)
# 	labels.append(cfg.end_flg)
	return labels

def get_char_indies(label):
	
	chars = list(label)
	if label in cfg.init_dic:
		chars = [label]
	labels = warp_sil_labels(chars)
	indies = map(lambda l: cfg.c2i(l), labels)
	
	return indies

def get_word_indies(label):
	labels = warp_sil_labels(label)
	
# 	#do not need begin and end flags
# 	labels = labels[1:-1]
	indies = map(lambda l: cfg.w2i(l), labels)
	return indies
	
def mfbank(x, sampling_rate):
	mfc = mfcc(x, sampling_rate, nfilt=40, numcep=40)
# 	d_mfc = delta(mfc, N=2)
# 	print mfc.shape
# 	print d_mfc.shape
# 	feat = np.hstack([mfc, d_mfc])
	feat = mfc
	return feat

def mfbank80(x, sampling_rate):
	mfc = mfcc(x, sampling_rate, nfilt=80, numcep=80)
# 	d_mfc = delta(mfc, N=2)
# 	print mfc.shape
# 	print d_mfc.shape
# 	feat = np.hstack([mfc, d_mfc])
	feat = mfc
	return feat

def mfcc10(x, sampling_rate, **arg):
	return mfcc(x, sampling_rate, nfft=1024, winlen=0.04, winstep=0.02, nfilt=49, numcep=10)

def mfcc40(x, sampling_rate, **arg):
	return mfcc(x, sampling_rate, nfft=1024, winlen=0.04, winstep=0.02, nfilt=49, numcep=40)

def mfcc40s(x, sampling_rate, **arg):
	return mfcc(x, sampling_rate, nfft=512, nfilt=49, numcep=40)

def logfbank40(x, sampling_rate, **arg):
	return logfbank(x, sampling_rate, nfilt=40, lowfreq=300, highfreq=3000)
def logfbank80(x, sampling_rate, **arg):
	return logfbank(x, sampling_rate, nfilt=80, lowfreq=300, highfreq=3000)

def ssc40(x, sampling_rate, **arg):
	return ssc(x, sampling_rate, nfilt=40)

def rawwav(x, sample_rate, **arg):
	
	return x

def zcr(x, sampling_rate, **arg):
	'''
	new features added by adam
	'''	
	winlen = 0.025
	winstep = 0.01
	wlen = len(x)
	frameSize =  int(winlen * sampling_rate)
	step = int(winstep * sampling_rate)
	frameNum = int(math.ceil(wlen/step)) - 1
	
	zerocr = np.zeros((frameNum, 1))

	#method 3
	def createframe(i):
		curFrame = x[np.arange(i * step, min(i * step + frameSize, wlen))]
		curFrame = curFrame - np.mean(curFrame)
		diffs = np.diff(curFrame > 0)
		nonzeros = np.nonzero(diffs)
		return len(nonzeros[0])
 	
	zerocr=[createframe(i) for i in range(frameNum)]	
# 

	
	#method 1
# 	for i in range(frameNum):
# 		curFrame = x[np.arange(i * step, min(i * step + frameSize, wlen))]
# 		curFrame = curFrame - np.mean(curFrame)
# 		zerocr[i]= sum(curFrame[0:-1] * curFrame[1::] <= 0)
	
# 	method 2
#  	for i in range(frameNum):
#  		curFrame = x[np.arange(i * step, min(i * step + frameSize, wlen))]
#  		curFrame = curFrame - np.mean(curFrame)
#  		diffs = np.diff(curFrame > 0)
#  		nonzeros = np.nonzero(diffs)
#  		zerocr[i] = len(nonzeros[0])

#	  
		
# 	framerate = sampling_rate
#  	
# 	time = np.arange(0, len(x)) * (1.0 / sampling_rate)
# 	pl.subplot(211)
# 	pl.plot(time, x)
# 	pl.ylabel("Amplitude")
# #	
# #	
# 	time2 = np.arange(0, len(zerocr)) * (1.0*(len(x)/len(zerocr)) / sampling_rate)
# 	pl.subplot(212)
# 	pl.plot(time2, zerocr)
# 	pl.ylabel("ZCR")
# 	pl.xlabel("time (seconds)")
# 	pl.show()

		
		
		
	
	return zerocr 

def get_STATKEngery(bpx,sampling_rate,winlen=0.03,win_shift=0.01):
	
	def TKEnergy_operator(t):
		y=np.cos(2*np.pi*t)
		yMinus=np.cos(2*np.pi*(t-1))
		yPlus=np.cos(2*np.pi*(t+1))
		TKX = y**2 - yMinus*yPlus;
		return TKX
	
	nfilt = len(bpx)
	wavelen = len(bpx[0])
	frameSize =  int(winlen * sampling_rate)
	step = int(win_shift * sampling_rate)
	frameNum = int(math.ceil(sampling_rate/step)) - 1
# 	
	E = np.zeros((nfilt,frameNum))
	for i in range(nfilt):
		for j in range(frameNum):
			curFrame = bpx[i][np.arange(j * step, min(j * step + frameSize, wavelen))]
			Et = [ TKEnergy_operator(xt) for xt in curFrame ]
			E[i][j] = sum(Et)
		
	return E




def erb_space(lowfreq, highfreq, nflit):

	earQ = 9.26449						
	minBW = 24.7  
	low = float(lowfreq)  
	high = float(highfreq)  
	N=float(nflit)  
	cf = -(earQ * minBW) + np.exp((np.arange(N+1)[1:]) * (-np.log(high + earQ * minBW) + np.log(low + earQ * minBW)) / (N)) * (high + earQ * minBW)  
	return cf  



def  get_gammatone_filterbank(fs,center_freqs,band_width = 1.0):

	erb = 6.23*(center_freqs/1000)**2 + 93.39*(center_freqs/1000)+28.5
	fs=float(fs)
	T = 1.0/fs
	B = 1.019*2*np.pi*erb
	arg = 2*center_freqs*np.pi*T
	vec = np.exp(2j*arg)
	A0 = T
	A2 = 0.0
	B0 = 1.0
	B1 = -2*np.cos(arg)/np.exp(B*T)
	B2 = np.exp(-2*B*T)
	
	rt_pos = np.sqrt(3 + 2**1.5)
	rt_neg = np.sqrt(3 - 2**1.5)
	common = -T * np.exp(-(B * T))
	k11 = np.cos(arg) + rt_pos * np.sin(arg)
	k12 = np.cos(arg) - rt_pos * np.sin(arg)
	k13 = np.cos(arg) + rt_neg * np.sin(arg)
	k14 = np.cos(arg) - rt_neg * np.sin(arg)
	A11 = common * k11
	A12 = common * k12
	A13 = common * k13
	A14 = common * k14
	gain_arg = np.exp(1j * arg - B * T)
	gain = np.abs(
		(vec - gain_arg * k11)
		* (vec - gain_arg * k12)
		* (vec - gain_arg * k13)
			* (vec - gain_arg * k14)
			* (  T * np.exp(B*T)
				/ (-1 / np.exp(B*T) + 1 + vec * (1 - np.exp(B*T)))
			  )**4
		 )	
	allfilts = np.ones_like(center_freqs)
	fcoefs = np.column_stack([
	 A0*allfilts, A11, A12, A13, A14, A2*allfilts,
	 B0*allfilts, B1, B2,
	 gain
	])
	return fcoefs
		 


def bandpass_signal(x, coefs=0.0):
	
	
	output = np.zeros((coefs[:,9].shape[0], x.shape[0]))	
	gain = coefs[:, 9]

	As1 = coefs[:, (0, 1, 5)]

	As2 = coefs[:, (0, 2, 5)]

	As3 = coefs[:, (0, 3, 5)]
  
	As4 = coefs[:, (0, 4, 5)]
 
	Bs = coefs[:, 6:9]
	
	
	for idx in range(0, coefs.shape[0]):
			y1 = signal.lfilter(As1[idx], Bs[idx], x)
			y2 = signal.lfilter(As2[idx], Bs[idx], y1)
			y3 = signal.lfilter(As3[idx], Bs[idx], y2)
			y4 = signal.lfilter(As4[idx], Bs[idx], y3)
	output[idx, :] = y4/gain[idx]
	
	return output

def TECCs(x, sample_rate, **arg):
	from scipy.fftpack import dct
	nfft=1024
	winlen=0.04
	winstep=0.02
	nfilt=49
	numcep=10
	lowfreq=300 
	highfreq=3000
	cfs = erb_space(lowfreq, highfreq,nfilt)
	filterbanks = get_gammatone_filterbank(sample_rate,cfs)
	bpx = bandpass_signal(x,filterbanks)
	feat = get_STATKEngery(bpx,sample_rate,winlen=0.03,win_shift=0.01)
	feat = np.log(feat)
	feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
	
	return feat

def foldwav(x, sample_rate, **arg):
	return np.reshape(x, (-1, 1600))


def integration(x, sample_rate, **arg):
	'''
	integration 34 dimension, including zcr, Energy, Entropy of Energy, Spectral Centroid, MFCCs, and so on.
	see:https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
	'''
	feats = audioFeatureExtraction.stFeatureExtraction(x, sample_rate, 0.025*sample_rate, 0.01*sample_rate);
	return feats.T

def get_features(data, name, sampling_rate, **arg):
	names = []
	outs = []
	if name.startswith("(") and name.endswith(")"):
		print "combined feature:" + name
		list_name = name[1:-1].split(',')
		names = map(lambda x:x.strip(), list_name)
	else:
		print "call {}() func".format(name)
		names = [name]
	max_dim_len = 0
	min_dim_len = sys.maxint
	for name in names:
		index = name.rfind("-")
		in_data = data
		if index > -1:
			sr = int(name[index + 1:])
			name = name[:index]
			if sampling_rate != sr:
				sampling_rate = sr
				in_data = down_sample(data, sampling_rate / float(cfg.sampling_rate))
		
		output = map(lambda x : eval(name)(x, sampling_rate), in_data, **arg)
		output = np.array(output)
		print output.shape
		outs.append(output)
		if len(output.shape) > max_dim_len:
			max_dim_len = len(output.shape)
		if len(output.shape) < min_dim_len:
			min_dim_len = len(output.shape)
# 	output = np.vstack(output)
# 	print output[0]
	if max_dim_len - min_dim_len > 1:
		print "the dim of feature is too large than small one"
		
	def add_dim(out):
		if len(out.shape) < max_dim_len:
			out = out.reshape(out.shape + (1,))
		return out
	
	outs = map(lambda out:add_dim(out), outs)

	features = np.concatenate(outs, axis=-1)
	print features.shape

	return features

def get_labels(labels, name, **arg):
	name = "label_" + name
	print "call {}() lable_func".format(name)
	if name == 'label_fname':
		output = arg['fnames']
	else:
		output = map(lambda x : eval(name)(x, **arg), labels)
	arr = np.array(output)
	print arr.shape

	return arr

def label_fname(label, **arg):
	return label

def label_name(label, **arg):
	return label

def label_simple(label, **arg):
	return np.int32(cfg.n2i(label))

def label_word(label, **arg):
	return np.int32(cfg.w2i(label))

def label_large(label, **arg):
	return np.int32(cfg.wl2i(label))

def label_word_seq(label, **arg):
	return get_word_indies(label)

def label_char_seq(label, **arg):
	return get_char_indies(label)
# def pad_data(data):
# 	lengths = map(lambda x : len(x), data)
# 	L = max(lengths)
# 	npz_data = np.zeros((len(data), L, data[0].shape[1]))
# 	for i, x in enumerate(data):
# 		pad_x = x
# 		if len(x) > L:
# 			print "found overlength item!"
# 		for j in range(len(x), L):
# 			pad_x = np.row_stack(x, np.ones(x.shape[1]) * cfg.pad_flg_index)
# 		npz_data[i] = pad_x
# 	return npz_data

def make_file_name(head, type, *args):
	return head + make_file_trim(type, args)


	

def gen_features_and_labels(data, feature_names=[], label_names=[], 
							is_aggregated=True, is_normalization=True, 
							down_rate=1.0, outdir="../data/train/train"):

	sampling_rate = int(cfg.sampling_rate * down_rate)
	x, labels, fnames = read_wav_and_labels(data, is_normalization=is_normalization)
	data_list = {}
	if down_rate < 1.0:
		x = down_sample(x, down_rate)
# 	x = pad_data(x)
	if feature_names is not None:
		for name in feature_names:
			print "gen feature:{}".format(name)
			feature = get_features(x, name, sampling_rate)
			if is_normalization:
				feature = standardization(feature)
			data_list["x_" + name] = feature
			
	if label_names is not None:
		for name in label_names:
			print "gen label:{}".format(name)
			label = get_labels(labels, name, fnames=fnames)
			data_list["y_" + name] = label
	
	save_file = ""
	if is_aggregated:
		save_file = make_file_name(outdir, "npz", sampling_rate)
		np.savez(save_file, **data_list)
		print "save file:{}".format(save_file)
	else:
		for k, v in data_list.iteritems():
			if k.startswith("x_"):
				save_file = make_file_name(outdir + "_" + k, "npz", sampling_rate)
				np.savez(save_file, x=v)
			else:
				save_file = outdir + "_" + k + ".npz"
				np.savez(save_file, y=v)
			print "save file:{}".format(save_file)		
	
	print "sampling rate:{}".format(sampling_rate)
	print "processed data num:{}".format(len(x))
	print "completed gen feature:{}, label:{}".format(len(feature_names), len(label_names))

def gen_train_feature(feature_names=[], label_names=[], is_aggregated=True, is_normalization=True, down_rate=1.0, outlier_path='../data/outlier/orig/', data_dir="../data/"):
# 	train, val = load_data(data_dir)
# 	sampling_rate = int(cfg.sampling_rate * down_rate)
# 	x, y, y_c, y_w = get_wav(train, is_normalization=is_normalization)
# 	x_wav = np.array(x)
# 	
# 	if down_rate < 1.0:
# 		x = down_sample(x, down_rate)
# # 	x = pad_data(x)
# 	
# 	x = get_features(x, name, sampling_rate)
# 	
# 	v_x, v_y, v_y_c, v_y_w = get_wav(val)
# 	v_x_wav = np.array(v_x)
# # 	v_x = pad_data(v_x)
# 	if down_rate < 1.0:
# 		v_x = down_sample(v_x, down_rate)
# 	v_x = get_features(v_x, name, sampling_rate)
# 	
# 	
# 	np.savez(make_file_name("../data/train/train", "npz", sampling_rate, name), x_wav=x_wav,x = x, y = y, y_c = y_c, y_w = y_w)
# 	np.savez(make_file_name("../data/valid/valid", "npz", sampling_rate, name), x_wav=v_x_wav,x = v_x, y = v_y, y_c = v_y_c, y_w = v_y_w)
# 	print "sampling rate:{}".format(sampling_rate)
# 	print "feature shape:{}*{}".format(x[0].shape[0], x[0].shape[1])
	train, val = load_data(data_dir, outlier_path)
	data = train + val
	gen_features_and_labels(data, feature_names, label_names, is_aggregated, is_normalization, down_rate, "../data/train/train")
# 	gen_features_and_labels(val, feature_names, label_names, is_aggregated, is_normalization, down_rate, "../data/valid/valid")
	print "completed gen training data..."

def load_test_data(data_dir="../data/test/audio/"):
	paths = gen_input_paths(data_dir, file_ext_name=".wav")
	print "test data num:{}".format(len(paths))
	data = []
	for fname in paths:
		data.append((os.path.basename(fname), fname))
	return data
	
def gen_test_feature(feature_names, is_aggregated, is_normalization=True, down_rate=1.0, data_dir="../data/test/audio/"):
# 	paths = gen_input_paths(data_dir + "audio/", ".wav")
# 	print "test data num:{}".format(len(paths))
# 
# 	x = []
# 	names = []
# 	sampling_rate = int(cfg.sampling_rate * down_rate)
# 	for i, fname in enumerate(paths):
# 		_, wav = wavfile.read(fname)
# 		wav = pad_audio(wav)
# 		wav = wav.astype(np.float32) / np.iinfo(np.int16).max
# 		L = cfg.sampling_rate
# 
# 		if len(wav) > L:
# 			beg = np.random.randint(0, len(wav) - L)
# 		else:
# 			beg = 0
# 		wav = wav[beg: beg + L]
# 		names.append(os.path.basename(fname))
# 		x.append(wav)
# 		
# 		if i % 10000 == 0:
# 			print "read {} files".format(i)
# 	x_wav = np.array(x)
# 	print "completed read wav files, start extracting fearures..."
# 
# 	x = get_features(x, name, sampling_rate)
# 	
# 	np.savez(make_file_name(data_dir + "test", "npz", sampling_rate, name), x = x, x_wav=x_wav, names=names)
# 	print "sampling rate:{}".format(sampling_rate)
# 	print "feature shape:{}*{}".format(x[0].shape[0], x[0].shape[1])
# 	print "completed gen test data..."
	data = load_test_data(data_dir)
# 	data = data[:10]
	gen_features_and_labels(data, feature_names, ['name'], is_aggregated, is_normalization, down_rate, "../data/test/test")
	
	
# def get_test_data_from_files(root_path="../data/test/", filter_trim=None):
# # 	trim = get_file_trim(filter_trim = filter_trim)
# # 	print "process file type:" + trim
# # 	paths = gen_input_paths(root_path, trim)
# # 	x_list = []
# # 	name_list = []
# # 	
# # 	print "We'll load {} files...".format(len(paths))
# # 	for path in paths:
# # 		print "load data from:" + path
# # 		data = np.load(path)
# # 		x_list.append(data['x'])
# # 		name_list.extend(list(data['names']))
# # 	
# # 	x = np.vstack(x_list)
# 	
# 	return x, name_list



def gen_input_paths_and_names(root_path="../data/ext/", file_beg_name="", file_ext_name=".csv", mode='file'):
	paths = gen_input_paths(root_path, file_beg_name, file_ext_name, mode)
	names = map(lambda x: os.path.basename(x), paths)
	return paths, names

def get_file_trim(type="npz", filter_trim=None):
	return ".{}".format(type) if (filter_trim == "") or (filter_trim is None) else "_{}.{}".format(filter_trim, type)

def make_file_trim(type="npz", *args):
	name = ".{}".format(type)
	if args is not None and len(args) > 0:
		for arg in args[0]:
			name = "_" + str(arg) + name
	
	return name
	
	

def get_data_from_files(root_path, down_rate, feature_names=[], label_names=[], input_num=0, pad_id=0):
	
	sampling_rate = int(cfg.sampling_rate * down_rate)
	trim = get_file_trim(filter_trim = str(sampling_rate))
	print "process file type:" + trim
	paths = gen_input_paths(root_path, file_ext_name=trim)

	data_dict = {}
	print "We'll load {} files...".format(len(paths))
	for path in paths:
		data = np.load(path)
# 		data = data['arr_0']
		for name in feature_names:
			name = "x_" + name
			item = data[name]
			if name in data_dict:
				data_dict[name].append(item)
			else:
				data_dict[name] = [item]
			print "load x {}:{} from:{}".format(name, item.shape, path)
		
		for name in label_names:
			name = "y_" + name
			item = data[name]
			if name in data_dict:
				data_dict[name].append(item)
			else:
				data_dict[name] = [item]
			print "load y {}:{} from:{}".format(name, item.shape, path)
			
	for name in data_dict.keys():
		data_dict[name] = concate_datas(data_dict[name], pad_id)
	
	ret = ()
	if input_num > 0:
		for name in feature_names:ret += (data_dict["x_" + name][:input_num], )
		for name in label_names:ret += (data_dict["y_" + name][:input_num], )
	else:
		for name in feature_names:ret += (data_dict["x_" + name], )
		for name in label_names:ret += (data_dict["y_" + name], )

		
	return ret


def concate_datas(data_list, pad_id=0):
	datas = []
	if check_type_consistency(data_list):
		if len(data_list[0].shape) > 1:
			datas = np.vstack(data_list)
		else:
			datas = np.hstack(data_list)
	else:
		for data in data_list:
			datas.extend(list(data))
		
		if pad_id != None and data_util.isDigitType(datas):
			datas = sequence.pad_sequences(datas, dtype=np.int32, padding='post', value=pad_id)
	return datas

def check_type_consistency(data_list):
	for data in data_list:
		if not data_util.isDigitArrType(data):
			return False
	return True

def load_data_ext(data_dir, default_label, outlier_path=None, include_labels=None, exclude_labels=None):
	paths = gen_input_paths(data_dir, file_ext_name=".wav")
	samples = []
	for path in paths:
		index = path.rfind('-')
		label = default_label
		if index > -1:
			str = path[index + 1:-4]
			label = str_to_label(str)
			#skip files with excluded label
		if exclude_labels is not None and label in exclude_labels:
			continue
		#skip files without included label
		if include_labels is not None and label not in include_labels:
			continue
		samples.append((label, path))
	if outlier_path is not None:
		outlier_set = get_outlier_set(outlier_path)
		samples = filter_outlier_files(outlier_set, samples)

	return samples		

def gen_ext_feature(id, feature_names, label_names, is_aggregated, is_normalization, 
					down_rate=1.0, default_label=cfg.sil_flg, outlier_path=None, outdir="../data/train/"):
	'''
	:param id: ext fold id, which stores your extension data, should be a number. For example, if your data is in 
				  dir:'data/train/ext1', you should set id = 1
	:param feature_names: features' names which would be extracted. 
	:param label_names:data labels, as 'Y' used in your model
	
	'''

	data = load_data_ext("../data/train/ext{}/".format(id), default_label, outlier_path=outlier_path)
	gen_features_and_labels(data, feature_names, label_names, is_aggregated, is_normalization, down_rate, "{}train_ext{}".format(outdir, id))
	print "completed gen ext \"{}\" feature of {} files from ext{}.".format(default_label, len(data), id)
	
def gen_feature(path, outdir, feature_names, label_names, is_aggregated, is_normalization, down_rate=1.0, default_label=cfg.sil_flg, outlier_path=None):	
	data = load_data_ext(path, default_label, outlier_path)
	gen_features_and_labels(data, feature_names, label_names, is_aggregated, is_normalization, down_rate, outdir)
	print "completed gen ext \"{}\" feature of {} files from {}.".format(default_label, len(data), path)
	



def mix_white_noise(data, rate=1.0):
	return merge_noise(data, rate, 'white_noise')

def mix_doing_the_dishes_noise(data, rate=1.0):
	return merge_noise(data, rate, 'doing_the_dishes')

def mix_exercise_bike_noise(data, rate=1.0):
	return merge_noise(data, rate, 'exercise_bike')

def mix_running_tap_noise(data, rate=1.0):
	return merge_noise(data, rate, 'running_tap')

def mix_real_noise(data, rate=1.0):
	return merge_noise(data, rate, 'real')

def mix_pink_noise(data, rate=1.0):
	return merge_noise(data, rate, 'pink_noise')

def mix_dude_miaowing_noise(data, rate=1.0):
	return merge_noise(data, rate, 'dude_miaowing')

def merge_noise(data, rate, name):
	noise_set = noise_dict[name]
	return mix_noise(data, rate, noise_set)

def mix_noise(data, rate=0.005, noise_set=noise_array):
# 	L = data.shape[-1]
	axis = len(data.shape) - 1
	num = 1 if axis <= 0 else data.shape[0]
	noises = None
	if num > 1:
		noises = np.vstack(random.sample(noise_set, num))
	else:
		noises = random.sample(noise_set, num)[0]
	return data + noises * rate


def vtlp(data, rate=1.0):
	return data

def shift(data, rate=0.1):
	'''
	shift wav data, if rate > 0, the data sequence will be shift right rate*len(data), 
	or else will be shift left
	'''
	L = data.shape[-1]
	offset = int(L * rate)
	
	return np.roll(data, offset, axis=len(data.shape) - 1)


def stretch(data, rate=1.0, input_length=cfg.sampling_rate):
	'''
	stretching wav data, if rate > 1, the frequency will higher than before, else lower than before
	'''
	if rate < 1.1 and rate > 1:
		rate = 1.2
	elif rate > 0.9 and rate < 1:
		rate = 0.8
	
	data = librosa.effects.time_stretch(data, rate)
	if len(data)>input_length:
		data = data[:input_length]
	else:
		data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
	
	return data

def shift_pitch(data, rate=30):
	'''
	rate = [-60, 60]
	'''

# 	y, sr = librosa.load(input_file, 16000)
	sr = cfg.sampling_rate
 	y = data
 	y = y.astype(np.float32)


	y_harm = librosa.effects.harmonic(y)


	# Just track the pitches associated with high magnitude
	tuning = librosa.estimate_tuning(y=y_harm, sr=sr)

# 	print('{:+0.2f} cents'.format(100 * tuning))
# 	print('Applying pitch-correction of {:+0.2f} cents'.format(-100 * tuning))
	y_tuned = librosa.effects.pitch_shift(y, sr, tuning * rate)
# 	y_tuned = librosa.effects.pitch_shift(y, sr, -tuning * 40)
# 	y_tuned = y_tuned.astype(np.int16)
	return y_tuned

def down_sample(data, rate=0.5):
	if rate != 1:
		return map(lambda x : librosa.resample(x, cfg.sampling_rate, int(cfg.sampling_rate * rate)), data)
	return data

def sampling_from_mix_gaussians(w, means, vars):
	N = len(w)
	w = np.array(w)
	index = np.random.choice(np.arange(N), p=w)
	sample = random.gauss(means[index], vars[index])
	return sample

def random_call(data, ops):
	
	out = data
	for op in ops:
		name = op[0]
		
		val = random.uniform(0, 1)
		
		if val < op[1]:
			rate = 0
			if len(op[2]) == 2:
				w = [1.0]
				means = [(op[2][0] + op[2][1]) / 2.0]
				vars = [abs(op[2][1] - op[2][0]) / 2.0]
				rate = sampling_from_mix_gaussians(w, means, vars)
			elif len(op[2]) == 3:
				w = [0.25, 0.25, 0.25, 0.25]
				means = [op[2][0] * 0.8, op[2][0] * 0.6, op[2][2] * 0.6, op[2][2] * 0.8]
				vars = [op[2][0] * 0.1, op[2][0] * 0.12, op[2][2] * 0.12, op[2][2] * 0.1]
				rate = sampling_from_mix_gaussians(w, means, vars)

			out = eval(name)(out, rate)
	return out

def label_to_str(label):
	str = label
	if label == cfg.sil_flg:
		str = cfg.sil_flg_str
	if label == cfg.unk_flg:
		str = cfg.unk_flg_str
	return str

def str_to_label(str):
	label = str
	if label == cfg.sil_flg_str:
		label = cfg.sil_flg
	if label == cfg.unk_flg_str:
		label = cfg.unk_flg
	return label


		
def augmentation(data, outdir='../data/train/ext14/', 
				ops=[("mix_noise", 0.99, (1.0, 2.0)), ("shift_pitch", 1.0, (-6, 0, 6)), ("shift", 0.1, (-0.03, 0.0)), ("stretch", 0.3, (0.85, 1, 1.25))]):
	
	make_if_not_exist(outdir)

	print "Total items:{}".format(len(data))
	cnt = 1
	for (label, fname) in data:
# 		print fname
		_, wav = read_wav(fname, False)
		if len(wav) > cfg.sampling_rate:
			print "skip overlength file {}".format(fname)
			continue

		wav = wav.astype(np.float64)
		wav = random_call(wav, ops)
		fname = os.path.basename(fname)
		index = fname.find(".wav")
		label = label_to_str(label)
		
		if "-" not in fname:
			fname_out = fname[0:index] + "-" + label + ".wav"
			fname_out = outdir + fname_out
			wav = wav.astype(np.int16)
			
			id = 1
			while os.path.exists(fname_out):
				fname_out = fname[0:index] + "_" + str(id) +"-" + label + ".wav"
				fname_out = outdir + fname_out
				id += 1
				print id
				
			wavfile.write(fname_out, cfg.sampling_rate, wav)
			print "processed :{}-{}".format(cnt, fname_out)
			cnt += 1
			
			
# def remove_silence():
# 	v = VoiceActivityDetector(filename)



def gen_features_realtime(wavs, features, is_augment, is_normalization=True):
	
	if is_augment:
		wavs = augment_realtime(wavs)
	data_list = []
	for name in features:
		feature = get_features(wavs, name, cfg.sampling_rate)
		if is_normalization:
			feature = standardization(feature)
		data_list.append(feature)
	return data_list

def augment_realtime(wav_data, 
					ops=[("mix_noise", 0.99, (1.0, 2.0)), 
						("shift_pitch", 1.0, (-25, 0, 25)), 
						("shift", 0.8, (-0.15, 0, 0.15)), 
						("stretch", 0.3, (0.85, 1, 1.25))]):
	wav_list = map(lambda wav : random_call(wav, ops), wav_data)
	return np.array(wav_list)
	
def gen_augmentation_data(outdir, ops, input_num=0, include_labels=None, load_train_data=True, exts=None):
	data = []
	if load_train_data:
		train, val = load_data("../data/", '../data/outlier/orig/', include_labels)
		data = train + val
	if exts is not None:
		for info in exts:
			data = data + load_data_ext(info[0], info[1],  include_labels=include_labels)
	if input_num > 0:
		indeies = np.random.random_integers(0, len(data), input_num)
		data = map(lambda x : data[x], indeies)
	augmentation(data, outdir, ops)
	
	
def extract_unk_from_extend(root_path="../data/testset/", outdir="../data/train/ext10/", input_num=0):
	
	
	if not os.path.exists(outdir):
		print "created dir:" + outdir
		os.mkdir(outdir)
	
	def check_txt(lines, vocab):
		org = vocab.wordset()
		for line in lines:
			word_set = set(line.split(" "))
			cross_set = word_set & org
			if len(cross_set) > 0:
				return False
		return True
			
	
	wav_files = gen_input_paths(root_path + "wav/", file_ext_name=".wav")
	txt_files = gen_input_paths(root_path + "txt/", file_ext_name=".txt")
	
	if input_num > 0:
		wav_files = wav_files[:input_num]
		txt_files = txt_files[:input_num]
	
	outputs = []
	for fdata, ftxt in zip(wav_files, txt_files):
		ft = open(ftxt, 'r')
		lines = ft.readlines()
		if check_txt(lines, cfg.voc_word):
			R, wav = wavfile.read(fdata)
			min_R = R * 2
			Len = len(wav)
			if Len > min_R:
				for i in xrange(R/2, Len - R, R):
					sdata = wav[i:i + R]
# 					print len(sdata)
					sdata = pad_audio(sdata, R)
# 					print len(sdata)
					outputs.append(sdata)
	print "collected {} wav splits".format(len(outputs))
	for i, wav in enumerate(outputs):
		out_wav = librosa.resample(wav, R, cfg.sampling_rate)
# 		print "{}->{}".format(len(wav), len(out_wav))
# 		out_wav = signal.resample(out_wav, cfg.sampling_rate)
		fname = outdir + "{}.wav".format(i)
		wavfile.write(fname, cfg.sampling_rate, out_wav)
		print "processed file:"	+ fname
					
	print "complated saved files to :{}".format(outdir)			


def shape_audio(wav, new_length):
	if len(wav) > new_length:
		wav = wav[0:cfg.sampling_rate]
	wav = pad_audio(wav)
	return wav

def display_wav_info(f):
	
	params = f.getparams()  
	nchannels, sampwidth, framerate, nframes = params[:4]
	print params
# 	str_data  = f.readframes(nframes)
# # 	print str_data
# 	f.close()
# 	data = audioop.lin2lin(str_data, 1, 2)
# 	wave_data = np.fromstring(data, dtype = np.int16)
# 	print len(wave_data)
	
def covert_8bitTo16bit(wav):
# 	wav = audioop.ulaw2lin(wav, 2)
	wav = audioop.alaw2lin(wav, 2)
	wave_data = np.fromstring(wav, dtype = np.int16)
	return wave_data
	
def extract_wordvoice(vocab=cfg.voc_word, sample_num=10000, ops=[("mix_noise", 0.59, (1.0, 2.0)), ("stretch", 0.59, (0.75, 1.35))],
					outdir="../data/train/ext13/", root_path="../data/voice/"
					):
	
	if not os.path.exists(outdir):
		print "created dir:" + outdir
		os.mkdir(outdir)
	
	folds = gen_input_paths(root_path, file_ext_name="", mode='fold')
	
	avg_num = sample_num / len(folds)
	for fold in folds:
		files = gen_input_paths(fold, file_ext_name=".wav")
		indeies = np.random.random_integers(0, len(files) - 1, avg_num)
		for i, index in enumerate(indeies):
			print "Processing file:" + files[index]
			label = os.path.basename(files[index])[:-4]
			label = label if vocab.w_in(label) else cfg.unk_flg_str
			print files[index]
			R = None
			wav = None
			try:
				f = wave.open(files[index],"rb")
				display_wav_info(f)
				R, wav = wavfile.read(files[index])
			except :
				print "Encounter a file format issue, skip file:" + files[index]
				continue
			fname = "{}_{}-{}.wav".format(fold[-1], i, label)
				
# 			wavfile.write("../data/test_old.wav", R, wav)
			
			if wav.dtype == np.uint8:
				wav = covert_8bitTo16bit(wav)
	
# 			wavfile.write("../data/test_new1.wav", R, wav)
			if R != cfg.sampling_rate:
				print len(wav)
				wav = librosa.resample(wav, R, cfg.sampling_rate)
# 				wav = wav + 128
# 				wavfile.write("../data/test_new2.wav", cfg.sampling_rate, wav)
				print len(wav)
			wav = shape_audio(wav, cfg.sampling_rate)
			print len(wav)
# 			wav = random_call(wav, ops)
# 			wavfile.write("../data/test_new3.wav", cfg.sampling_rate, wav)
			fname = outdir + fname
			print "extracted file to:{}".format(fname)
			wavfile.write(fname, cfg.sampling_rate, wav)
			print ""
			print ""

def mfcc_tf(wavs, sampling_rate, **arg):
	# Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
	spectrogram = contrib_audio.audio_spectrogram(
		wavs,
		window_size=640,
		stride=320,
		magnitude_squared=True)
	op_mfcc = contrib_audio.mfcc(
		spectrogram,
		sampling_rate,
		dct_coefficient_count=40)
	sess = arg['sess']
	data = sess.run(op_mfcc)

	return data

def extract_feature(root_path, feat_name, is_normalization = True):
	paths = gen_input_paths(root_path, file_ext_name='.wav')
	names = []
	data = []
	for path in paths:
		_, samples = wavfile.read(path)
		data.append(pad_audio(samples))
		names.append(path)
	feat_all = get_features(data, feat_name, cfg.sampling_rate)
	if is_normalization:
		feat_all = standardization(feat_all)
	return names, feat_all

def loc_fft(data, sample_rate):
	_, val = custom_fft(data, sample_rate)
	return val

def energy(data, sample_rate):
	data = data**2
	data = np.sum(data)
	return data
# 	return data.reshape(-1, 1)

def flatten_mfcc(data, sample_rate):
	data = mfcc(data, sample_rate)
	data = data.reshape((data.shape[0] * data.shape[1], ))
	return data
def speech_ratio(data, sample_rate):
	data = VAD.speech_ratio(data, sample_rate)
	return data

def make_new_name(olddir, outdir, label):
	name = os.path.basename(olddir)
	label = label_to_str(label)
	if name.rfind('-') == -1:
		name = "{}-{}.wav".format(name[0:-4], label)
	olddir = outdir + name
	return olddir

def detect_outlier(root_path, label, n_components=3, contamination=0.005, outdir="../data/outlier/"):
	from sklearn.covariance import EllipticEnvelope
	from sklearn.svm import OneClassSVM
	from sklearn.ensemble import IsolationForest
	
	names, x_train = extract_feature(root_path, 'logspecgram-8000')
	if len(x_train.shape) == 3:
		x_train = np.sum(x_train, 2)
	
	print "input {} shape:{}".format(label, x_train.shape)
	pca = PCA(n_components=n_components)
	x_train = pca.fit_transform(x_train)
	
# 	clf = EllipticEnvelope(contamination=contamination)
	clf = IsolationForest(contamination=contamination, max_samples=x_train.shape[0], random_state=np.random.RandomState(42))
	
	clf.fit(x_train)
	y_pred = clf.predict(x_train)
	
	indeies_inlier = np.argwhere(y_pred==1)
	indeies_outlier = np.argwhere(y_pred==-1)
	indeies_inlier = np.squeeze(indeies_inlier)
	indeies_outlier = np.squeeze(indeies_outlier)
	print "outliers' number:{}".format(len(indeies_outlier))
	
	plt.figure()
	plt.title(label)
	plt.plot(x_train[indeies_inlier, 0], x_train[indeies_inlier, 1], 'bx')
	plt.plot(x_train[indeies_outlier, 0], x_train[indeies_outlier, 1], 'ro')
	plt.savefig("{}{}.jpg".format(outdir, label))
	plt.show()
	

	
	map(lambda x : shutil.copyfile(names[x], make_new_name(names[x], outdir, label)), indeies_outlier)
	
	print "Saved {} {} files to {}".format(len(indeies_outlier), label, outdir)

def get_outlier_set(outliers_path="../data/outlier/"):
	data = load_data_ext(outliers_path, cfg.unk_flg)
	outlier_set = []
	for label, fname in data:
		key = make_set_name(label, fname)
		outlier_set.append(key)
	return set(outlier_set)

def make_set_name(label, fname):
	name = os.path.basename(fname)
	index = name.rfind('-')
	if index != -1:
		name = name[:index] + '.wav'
	return name + "-" + label
def filter_outlier_files(outlier_set, data):
	out_data = []
	for label, fname in data:
		key = make_set_name(label, fname)
		if key not in outlier_set:
			out_data.append((label, fname))
		else:
			print "deleted outlier file:{}-{}".format(fname, label)
	return out_data


def detect_training_data(word_set):
	paths = gen_input_paths('../data/train/audio/', file_ext_name="", mode='fold')
	for path in paths:
		ind = path.rfind('/')
		word = path[ind + 1:]
		if word in word_set:
			print "detecting {} datas".format(word)
			detect_outlier(path, word)
			
def add_word_label(root_path, label, trim=".wav", allow_muti_labels=False):
	paths = gen_input_paths(root_path, file_ext_name=trim)
	for path in paths:
		
		
		if not allow_muti_labels:
			label_index = path.rfind('-')
			if label_index < 0:
				new_name = path[0:-len(trim)] + "-" + label + trim
				os.rename(path, new_name)
				print "rename: {} --> {}".format(path, new_name)
		else:
			new_name = path[0:-len(trim)] + "-" + label + trim
			os.rename(path, new_name)
			print "rename: {} --> {}".format(path, new_name)
			
def copy_suspect_data(suspect_id_dir='../data/outlier/', target_dir='../data/outlier/temp/', datadir="../data/train/audio"):
	paths = gen_input_paths(suspect_id_dir, file_ext_name='.wav')
	ids = set([])
	
	def get_id_from_path(path):
		fname = os.path.basename(path)
		index = fname.rfind('-')
		id = fname[0:-4]
		if index >= 0:
			id = fname[0:index]
		return id
	
	for path in paths:
		id = get_id_from_path(path)
		ids.add(id)
	
	paths = gen_input_paths(datadir, file_ext_name='', mode='fold')
	for path in paths:
		if not path.endswith("_"):
			inner_paths = gen_input_paths(path, file_ext_name='.wav')
			for fpath in inner_paths:
				id = get_id_from_path(fpath)
				if id in ids:
					label_index = path.rfind('/') + 1
					label = path[label_index:]
					dst_name = make_new_name(fpath, target_dir, label)
					shutil.copy(fpath, dst_name)
					print "copy suspect data: {} --> {}".format(fpath, dst_name)
	
				
	
def detect_signal_word(word, n_components=3, contamination=0.005):
	detect_outlier('../data/train/audio/{}'.format(word), word, n_components, contamination)
	
def copy_and_remane(output_num=0, start_index=0, threshold=0.0,
				name_dict_path='../sub/prediction_tf_dscnn_ml_all4_88727_85.csv', 
				orig_dir='../data/test/audio/', 
				target_dir='../data/train/ext22/', 
				extract_labels=None):
	if not os.path.exists(target_dir):
		print "created dir:" + target_dir
		os.mkdir(target_dir)

	df = pd.read_csv(name_dict_path)
	name_dict = {}
	for i in range(len(df)):
		if df.at[i, 'confidence'] >= threshold:
			name_dict[df.at[i, 'fname']] = df.at[i, 'plabel']
	paths = gen_input_paths(orig_dir, file_ext_name=".wav")
	
	cnt = 0
	
	paths = paths[start_index:]
	for path in paths:
		name = os.path.basename(path)
		label = name_dict.get(name, None)
		if label is None:
			continue
		if extract_labels is not None and label not in extract_labels:
			continue
		dst_name = make_new_name(path, target_dir, label)
		shutil.copy(path, dst_name)
		cnt += 1
		print "copy test data {}: {} --> {}".format(cnt, path, dst_name)
		
		if output_num > 0 and cnt >= output_num:
			break
def copy_special_labels(dirs, include_labels=None):
	for orig_dir, target_dir in dirs:
		make_if_not_exist(target_dir)
		datas = load_data_ext(orig_dir, cfg.unk_flg, include_labels=include_labels)
		for label, fname in datas:
			new_fname = make_new_name(fname, target_dir, label)
# 			new_fname = make_unique_name(new_fname, label)
			shutil.copy(fname, new_fname)
			print "copy = {} = data: {} --> {}".format(label, fname, new_fname)
			
def count_label_info(input_dir, outdir=None):

	samples = load_data_ext(input_dir, cfg.unk_flg)
	dict_data = {}
	for sample in samples:
		label = sample[0]
		path = sample[1]

		if label in dict_data:
			dict_data[label] = dict_data[label] + 1
		else:
			dict_data[label] = 1
	cls_num = len(dict_data)
	
	labels = []
	numbers = []
	
	for label, num in dict_data.iteritems():
		labels.append(label)
		numbers.append(num)
	df = pd.DataFrame({"label":labels, "number":numbers})
	df = df.sort_values(by = 'number',axis = 0, ascending = True) 
	print df
	if outdir is not None:
		df.to_csv(outdir, index=False)
		print 'save info to:' + outdir


def make_unique_name(fname, label):
	id = 1
	index = fname.rfind("-")
	fname_out = fname
	while os.path.exists(fname_out):
		
		if index < 0:
			fname_out = fname[0:-len('.wav')] + "_" + str(id) +"-" + label + ".wav"
		else:
			fname_out = fname[0:index] + "_" + str(id) +"-" + label + ".wav"

		id += 1
# 		print id
	return fname_out
		
def gen_ext_data(input_dir, outdir, gen_num, include_labels=None, exclude_labels=None, ops=[("mix_noise", 0.8, (0.5, 1.5)), ("shift_pitch", 0.95, (-45, 0, 15)), 
										("shift", 0.1, (-0.02, 0.0)), ("stretch", 0.1, (0.90, 1, 1.25))]):
	samples = load_data_ext(input_dir, cfg.unk_flg, outlier_path=None, include_labels=include_labels, exclude_labels=exclude_labels)
	dict_data = {}
	for sample in samples:
		label = label_to_str(sample[0])
		path = sample[1]
		
		if label in dict_data:
			dict_data[label].append(path)
		else:
			dict_data[label] = [path]
	cls_num = len(dict_data)
	avg_num = int(math.ceil(gen_num / float(cls_num)))
	
	print "Input num:{}, Total num:{}, label num:{}, Each label:{}".format(gen_num, cls_num * avg_num, cls_num, avg_num)
	
	make_if_not_exist(outdir)
	cls_cnt = 0
	for label, datas in dict_data.iteritems():
		max = len(datas) - 1
		indies = np.random.random_integers(0, max, avg_num)
		cls_cnt += 1
		cnt = 1
		for i in indies:
			fname = datas[i]
			sr, wav = read_wav(fname, False)
			if len(wav) > cfg.sampling_rate:
				print "skip overlength file {}".format(fname)
				continue
			wav = wav.astype(np.float64)
			wav = random_call(wav, ops)
			
			wav = wav.astype(np.int16)
			new_fname = make_new_name(fname, outdir, label)
			new_fname = make_unique_name(new_fname, label)
			wavfile.write(new_fname, sr, wav)
			print "label:{}, class index:{}, processed :{}-{}".format(label, cls_cnt, cnt, new_fname)
			cnt += 1
			
# 	for path in paths
def move_ext_data(tar_id, include_label, fold_ids=[1, 2, 23, 24]):
	
	
	dirs = []
	for org in fold_ids:
		dirs.append(('../data/train/ext{}/'.format(org), '../data/train/ext{}/'.format(tar_id)))
	copy_special_labels(dirs, [include_label])

def copy_files_accord_date(orig_path, dist_path, file_ext_name=".wav", end_time='2018-01-09 19:57:35'):
	paths = gen_input_paths(orig_path, file_ext_name=file_ext_name)
# 	copy_files = []
	cnt = 0
	if end_time is not None:
		end = time.mktime(time.strptime(end_time, '%Y-%m-%d %H:%M:%S'))
		for path in paths:
			mtime = os.path.getmtime(path)
			
			if mtime > end:
				new_fname = dist_path + os.path.basename(path)
				shutil.copy(path, new_fname)
				cnt += 1
				print "copy file {}:{} --> {}".format(cnt, path, new_fname)
			else:
				print datetime.datetime.fromtimestamp(mtime)

def find_label_and_id(path):
	fname = os.path.basename(path)
	index = fname.rfind("-")
	label = cfg.unk_flg
	id = fname[0:-4]
	if index > -1:
		label = fname[index + 1:-4]
		id = fname[0:index]
	return label, id
def compare_and_move(truth_path, src_path, dst_path, is_copy=True):
	paths = gen_input_paths(src_path, file_ext_name=".wav")
	name_dict = {}
	for fname in paths:
		label, id = find_label_and_id(fname)
		name_dict[id] = label
	cnt = 0
	paths = gen_input_paths(truth_path, file_ext_name=".wav")
	for fname in paths:
		label_truth, id = find_label_and_id(fname)
		if id in name_dict and label_truth != name_dict[id]:
			orig_name = src_path + id + '-' + name_dict[id] + '.wav'
			dst_name = dst_path + id + '-' + label_truth + '.wav'
			cnt += 1
			if is_copy:
				shutil.copy(orig_name, dst_name)
				print "copy file {}:{} --> {}".format(cnt, orig_name, dst_name)
			else:
				shutil.move(orig_name, dst_name)
				print "move file {}:{} --> {}".format(cnt, orig_name, dst_name)
				
def copy_duplicated_files(src_path, dst_path, is_copy=True):
	paths = gen_input_paths(src_path, file_ext_name=".wav")
	name_dict = {}
	for fname in paths:
		label, id = find_label_and_id(fname)
		if id in name_dict:
			name_dict[id].append(label)
		else:
			name_dict[id] = [label]
	
	cnt = 0
	for id, labels in name_dict.iteritems():
		if len(labels) > 1:
			for label in labels:
				cnt += 1
				orig_name = src_path + id + '-' + label + '.wav'
				dst_name = dst_path + id + '-' + label + '.wav'
				if is_copy:
					shutil.copy2(orig_name, dst_name)
					print "copy file {}:{} --> {}".format(cnt, orig_name, dst_name)
				else:
					shutil.move(orig_name, dst_name)
					print "move file {}:{} --> {}".format(cnt, orig_name, dst_name)
					
def compare_train_and_test_data(outdir='../data/train/ext35/'):
	import os,sys,md5
	train, val = load_data('../data/', outlier_path=None)
	data_train = train + val
	data_test = load_test_data("../data/test/audio/")
	cnt = 0
	index = 0
	for (label, fname) in data_train:
		for (label_t, fname_t) in data_test:
			f1 = open(fname, 'r')
			f2 = open(fname_t, 'r')
			index += 1
			if index % 100000 == 0:
				print "now is {}".format(index)
			if md5.new(f1.read()).digest() == md5.new(f2.read()).digest():
				cnt += 1
				print "{}={}".format(fname, fname_t)
				new_name = make_new_name(fname_t, outdir, label)
				shutil.copy2(fname_t, new_name)
				print "copy file {}:{} --> {}".format(cnt, fname_t, new_name)
				
def copy_error_files(df_pre, outdir='../data/train/ext33/'):
# 	df_wrong = df_pre[df_pre['plabel'] != df_pre['truth']]
	cnt = 0
	for i in range(len(df_pre)):
		plabel = df_pre.at[i, 'plabel']
		truth = df_pre.at[i, 'truth']
		if plabel != truth:
			fname = df_pre.at[i, 'fname']
			name = os.path.basename(fname)
			index = name.find('-')
			if index > -1:
				name = name[0:index]
			cnt += 1
			new_fname = '{}{}-{}-{}.wav'.format(outdir, name, label_to_str(plabel), label_to_str(truth))
			shutil.copy(fname, new_fname)
			print "copy wrong data {}: {} --> {}".format(cnt, fname, new_fname)
			
def test():
	'''
	label name: corresponding to 'truth Y'
	options:['simple', 'word', 'large', 'word_seq', 'char_seq', 'name', 'fname']
	:param	simple:the simplest label, only include 12 labels, cfg.POSSIBLE_LABELS + [<SIL>,<UNK>]
	:param	word:include all word appearing in the training set
	:param  large:include all words appearing in the whole data set
	:param	word_seq:word sequence for seq2seq model, which will warp each item with <SIL>, 'off' -> <SIL> off <SIL>
	:param	char_seq:char sequence for seq2seq model, split word in word_seq by space, 'off' -> <SIL> o f f <SIL>
	
	feature name: corresponding to input X with shape(Time, feature dimension), here Time is number of processing windows
	options:['mfcc', 'mfcc10', 'mfcc40', 'mfcc40s', 'logfbank', 'logfbank40', 'logfbank80', "logspecgram", 'rawwav', 'foldwav']
	The feature name in npz file will be:'x_feautureName', e.g. if input feature names:['mfcc10', 'logfbank'], 
		extructed features will be store in corresponding keys:'x_mfcc10', 'x_logfbank'
	
	Specify sampling rate: using feature name + "-" + smapling rate to specify the sampling rate toward this feature.
							e.g. gen_ext_feature(feature_names=['logfbank', 'mfcc40s', 'logspecgram-8000'], label_names=['word'], down_rate=1.0)
							here, input feature names is: [logfbank, mfcc40s, logspecgram].The sample rate will be setted to 16000hz because the down_rate=1.0,
							However, the feature logspecgram is appended '-8000' which means its sampling rate will be specified as 8000hz.
	:param	mfcc:MFCC, shape=Time * 13 (large window size:0.04)
	:param	mfcc10:MFCC, shape=Time * 10 (large window size:0.04)
	:param	mfcc40:MFCC, shape=Time * 40 (large window size:0.04)
	:param	mfcc40s:MFCC, shape=Time * 40 (small window size:0.025)
	:param	logfbank:logfbank, shape=Time * 26 (small window size:0.025)
	:param	logfbank40:logfbank, shape=Time * 40 (small window size:0.025)
	:param	logfbank80:logfbank, shape=Time * 80 (small window size:0.025)
	:param	logspecgram:logspecgram, shape=Time * 81 or 161, different from sampling rate (small window size:0.02)
	:param	rawwav:raw wav data, shape=total sampling points * 1
	:param	foldwav:fold raw wav data, shape=(total sampling points / fold number) * fold number
	:param  zcr:zero-crossing rate (to do)
	:param	TECCs:teager energy cepstrum coefficients (to do)
	:param  TEMFCC:teager-based Mel-Frequency cepstral coefficients (to do)
	
	Feature concatenate: gen_XXX_feature supports concatenate into one feature vector, input features like this:
						(name1,...,nameN). 
						Notice:
							1.The output feature name in npz file is '(name1,...,nameN)_x';
							2.features will be concatenated along with the last dimension, 
								thus, they must have the same shape except the last dimension.
							3.mfcc10:(Time, 10);logfbank:(Time, 26), so (mfcc10, logfbank):(Time, 36)
	
	Feature extracting:
	training data set:call function gen_train_feature
	test data set:call function gen_test_feature
	ext data set:
	1.make sure your extension data is set in "data/train/ext{id}/", here id should be a number.
	2.call function gen_ext_feature
	
	Add new feature extractors:
	Please refer to function logspecgram, and use your function name as the feature name when call gen_XXX_feature function.
	For example, if you would add new feature zero-crossing rate, please follow below steps:
	1.implemented a new feature extractor function named zcr, this function should keep the same arguments list as 
	  the extant feature functions: zcr(x, sampling_rate, **arg). the x is raw wav data, sampling_rate represents sample rate,
	  and **arg is your private parameters.
	2.make a new extXX fold under data/train/, like data/train/ext1, and put a few of test wav files in this fold.
	3.call gen_ext_feature function in test() function
	4.after that, the zcr feature will be store in file ext1_XXX.npz, use numpy.load read your data, and the key is "x_zcr"
	5.check if your feature data is wrong or no.
	'''
# 	ref_lable_name = 
# 	ref_feature_name = 

# 	func = logfbank
# 	print make_file_name("../data/train/train", "npz", 16000, "logspecgram")
	feat_names = ["rawwav", "logspecgram-8000", 'mfcc40s', 'zcr', 'ssc40', 'integration']
# 	feat_names = ['integration']
	label_names = ['simple', 'word', 'large', 'fname', 'name']
	is_normalization = True
	is_aggregated = True
	down_rate=cfg.down_rate
# 	compare_train_and_test_data()
# 	copy_duplicated_files(src_path='../data/train/ext2/', dst_path='../data/train/ext4/', is_copy=False)
# 	compare_and_move(truth_path='../data/train/ext2/', src_path='../data/train/ext4/', dst_path='../data/train/ext34/', is_copy=False)
# 	copy_special_labels(dirs = [('../data/train/ext1/','../data/train/ext31/')], include_labels=[cfg.sil_flg])
# 	add_word_label("../data/train/ext1/", cfg.sil_flg_str)
	
# 	gen_ext_data('../data/train/ext3/', '../data/train/ext4/', 10)
# 	gen_augmentation_data('../data/train/ext6/', ops=[
# 					("mix_noise", 0.98, (1000, 8000)), 
# 					("shift_pitch", 0.95, (-5.5, 5.5)), 
# 					("shift", 0.1, (-0.03, 0.0))], 
# 					input_num=0)
# 	gen_ext_feature(0, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.sil_flg)
# 	gen_ext_data('../data/train/ext31/', '../data/train/ext35/', 1700, include_labels=None, exclude_labels=None,
# 				ops=[("mix_noise", 0.9, (1000, 5000)), 
# 					("shift_pitch", 0.95, (-4.5, 4.5)), 
# 					("shift", 0.1, (-0.03, 0.0))])
# 	gen_ext_data('../data/train/ext2/', '../data/train/ext5/', 5000, include_labels=None, exclude_labels=(cfg.voc_word.wordset() - set([cfg.unk_flg])),
# 				ops=[("mix_running_tap_noise", 0.9, (2000, 8000)), 
# 					("shift_pitch", 0.95, (-5, 5)), 
# 					("shift", 0.1, (-0.03, 0.0))])
# 	gen_ext_data('../data/train/ext2/', '../data/train/ext5/', 5000, include_labels=None, exclude_labels=(cfg.voc_word.wordset() - set([cfg.unk_flg])),
# 				ops=[("mix_pink_noise", 0.9, (2000, 8000)), 
# 					("shift_pitch", 0.95, (-5, 5)), 
# 					("shift", 0.1, (-0.03, 0.0))])
# 	gen_ext_data('../data/train/ext2/', '../data/train/ext5/', 5000, include_labels=None, exclude_labels=(cfg.voc_word.wordset() - set([cfg.unk_flg])),
# 				ops=[("mix_white_noise", 0.9, (2000, 8000)), 
# 					("shift_pitch", 0.95, (-5, 5)), 
# 					("shift", 0.1, (-0.03, 0.0))])
# 	gen_ext_data('../data/train/ext2/', '../data/train/ext5/', 5000, include_labels=None, exclude_labels=(cfg.voc_word.wordset() - set([cfg.unk_flg])),
# 				ops=[("mix_doing_the_dishes_noise", 0.9, (2000, 8000)), 
# 					("shift_pitch", 0.95, (-5, 5)), 
# 					("shift", 0.1, (-0.03, 0.0))])
# 	gen_ext_data('../data/train/ext2/', '../data/train/ext5/', 5000, include_labels=None, exclude_labels=(cfg.voc_word.wordset() - set([cfg.unk_flg])),
# 				ops=[("mix_exercise_bike_noise", 0.9, (2000, 8000)), 
# 					("shift_pitch", 0.95, (-5, 5)), 
# 					("shift", 0.1, (-0.03, 0.0))])
# 	
# 	gen_ext_data('../data/train/ext2/', '../data/train/ext5/', 25000, include_labels=None, exclude_labels=(cfg.voc_word.wordset() - set([cfg.unk_flg])),
# 				ops=[("mix_noise", 1.0, (500, 3000)), 
# 					("shift_pitch", 0.95, (-5, 5))])
# 	
# 	gen_ext_data('../data/train/ext4/', '../data/train/ext8/', 5000, include_labels=(cfg.voc_large.wordset() - cfg.voc_small.wordset()) | set([cfg.unk_flg, cfg.sil_flg]),
# 				ops=[("mix_noise", 1.0, (500, 3000)), 
# 					("shift_pitch", 0.95, (-5, 5))])
	
# 	gen_ext_data('../data/train/ext3/', '../data/train/ext4/', 20000, include_labels=None, exclude_labels=cfg.voc_word.wordset(),
# 				ops=[("mix_noise", 0.8, (0.5, 1.5)), ("shift_pitch", 0.95, (-15, 0, 15)), 
# 										("shift", 0.1, (-0.03, 0.0)), ("stretch", 0.0, (0.90, 1, 1.25))])
# 	gen_ext_data('../data/train/ext3/', '../data/train/ext5/', 20000, include_labels=None, exclude_labels=cfg.voc_word.wordset(),
# 				ops=[("mix_noise", 0.8, (0.5, 1.5)), ("shift_pitch", 0.95, (-25, 0, 10)), 
# 										("shift", 0.1, (-0.03, 0.0)), ("stretch", 0.0, (0.90, 1, 1.25))])
# 	gen_ext_data('../data/train/ext3/', '../data/train/ext5/', 20000, include_labels=None, exclude_labels=cfg.voc_word.wordset(),
# 				ops=[("mix_noise", 0.8, (0.5, 1.5)), ("shift_pitch", 0.95, (-8, 0, 8)), 
# 										("shift", 0.1, (-0.03, 0.0)), ("stretch", 0.0, (0.90, 1, 1.25))])
# 	
# 	gen_ext_data('../data/train/ext3/', '../data/train/ext20/', 2000, include_labels=None, exclude_labels=None,
# 				ops=[("mix_noise", 0.5, (0.8, 1.5)), ("shift_pitch", 0.75, (-7.5, 7.5)), 
# 										("shift", 0.1, (-0.03, 0.0)), ("stretch", 0.0, (0.90, 1, 1.25))])

# 	gen_ext_data('../data/train/ext8/', '../data/train/ext21/', 3000, include_labels=None, exclude_labels=None,
# 				ops=[("mix_noise", 0.5, (0.8, 1.5)), ("shift_pitch", 0.75, (-5.5, 5.5)), 
# 										("shift", 0.1, (-0.03, 0.0)), ("stretch", 0.0, (0.90, 1, 1.25))])
# 	gen_ext_data('../data/train/ext23/', '../data/train/ext25/', 5000, include_labels=None, exclude_labels=None,
# 				ops=[("mix_noise", 0.5, (0.8, 1.5)), ("shift_pitch", 0.75, (-5.5, 5.5)), 
# # 										("shift", 0.1, (-0.03, 0.0)), ("stretch", 0.0, (0.90, 1, 1.25))])
# 	gen_ext_data('../data/train/ext24/', '../data/train/ext25/', 5000, include_labels=None, exclude_labels=None,
# 				ops=[("mix_noise", 0.5, (0.8, 1.5)), ("shift_pitch", 0.75, (-5.5, 5.5)), 
# 										("shift", 0.1, (-0.03, 0.0)), ("stretch", 0.0, (0.90, 1, 1.25))])
# 	gen_ext_data('../data/train/ext30/', '../data/train/ext31/', 10, include_labels=None, exclude_labels=None,
# 				ops=[("mix_white_noise", 1.0, (4000, 5000))])
# 	gen_ext_data('../data/train/ext30/', '../data/train/ext31/', 10, include_labels=None, exclude_labels=None,
# 				ops=[("mix_pink_noise", 1.0, (5000, 6000))])
	
# 	gen_ext_data('../data/train/ext30/', '../data/train/ext31/', 10, include_labels=None, exclude_labels=None,
# 				ops=[("mix_running_tap_noise", 1.0, (7000, 8000))])
	
# 	add_word_label('../data/train/ext7/', cfg.unk_flg_str, ".wav", False)
# 	copy_files_accord_date('../data/train/ext30/', '../data/train/ext33/')
# 	copy_and_remane(output_num=0, start_index=0, threshold=0.0, 
# 				name_dict_path='../sub/prediction_tf_dscnn_ml_all3_91152_86.csv',
# 				extract_labels=["white"],
# # 				extract_labels=(cfg.voc_large.wordset() - cfg.voc_word.wordset() - set(['follow', 'backward', 'visual', 'learn', 'withone', 'forward'])), 
# 				target_dir='../data/train/ext31/')
# # 	copy_and_remane(output_num=0, start_index=0, threshold=0.94, extract_labels=['happy'], 
# # 				name_dict_path='../sub/prediction_tf_dscnn_ml_all4_85606.csv', target_dir='../data/train/ext31/')
# # 	copy_and_remane(output_num=200, extract_labels=['off'], target_dir='../data/train/ext33/')
# # 	copy_and_remane(output_num=200, extract_labels=['no'], target_dir='../data/train/ext33/')
# # 	copy_and_remane(output_num=200, extract_labels=['up'], target_dir='../data/train/ext33/')
# # 	copy_and_remane(output_num=200, extract_labels=['stop'], target_dir='../data/train/ext33/')
# # 	copy_and_remane(output_num=100, start_index=70000, extract_labels=[cfg.unk_flg], target_dir='../data/train/ext28/')
# # 	copy_suspect_data()
# 	count_label_info('../data/train/ext2/', '../data/train/ext2.csv')
# 	count_label_info('../data/train/ext30/', '../data/train/ext30.csv')
# 	count_label_info('../data/train/ext32/', '../data/train/ext32.csv')
# 	detect_training_data(cfg.POSSIBLE_LABELS)
# 	detect_signal_word('no', n_components=2, contamination=0.01)
# 	gen_feature('../data/outlier/no/', "../data/no_outlier", ['logspecgram-8000'], ['simple'], is_aggregated, is_normalization, default_label=cfg.unk_flg)
# 	gen_feature('../data/outlier/no_all/', "../data/no", ['logspecgram-8000'], ['simple'], is_aggregated, is_normalization, default_label='no', outlier_path='../data/outlier/no/')
# 	gen_feature('../data/train/audio/no/', "../data/outlier/no_test/no_test", ['logspecgram-8000'], ['name'], is_aggregated, is_normalization, default_label='no')
# 	add_word_label("../data/outlier/neg/", 'no')
# 	for i in range(15, 17):
# 		gen_augmentation_data('../data/train/ext{}/'.format(i), 0, cfg.voc_word.wordset(), True, exts=[("../data/train/ext1/", cfg.sil_flg),
# 															("../data/train/ext26/", cfg.unk_flg)])
# 	extract_wordvoice()
# 	extract_unk_from_extend(root_path="../data/trainset/", outdir="../data/train/ext11/", input_num=0)
# 	gen_augmentation_data()
# 	default_label = 'off'
# 	R, wav = wavfile.read("../data/train/audio/bed/0a7c2a8d_nohash_0.wav")
# 	out_wav = librosa.resample(wav, 16000, 8000)
# # 	out_wav = signal.resample(wav, 8000)
# 	print len(out_wav)
# 	wavfile.write("../data/train/0a7c2a8d_nohash_0.wav", 8000, out_wav)
# 	gen_train_feature(feat_names, label_names, is_aggregated, is_normalization, down_rate)
# 	gen_test_feature(feat_names, is_aggregated, is_normalization, down_rate)
# 	gen_ext_feature(0, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.sil_flg)
# 	gen_ext_feature(1, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.sil_flg, outdir="../data/train/Temp/")
# 	gen_ext_feature(2, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg, outdir="../data/train/Temp/")
# 	gen_ext_feature(3, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(4, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg, outdir="../data/valid/")
# 	gen_ext_feature(5, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(6, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(7, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(8, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg, outdir="../data/valid/")
# 	gen_ext_feature(10, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg, outdir="../data/train/Temp/")
# 	gen_ext_feature(9, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(11, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(12, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(13, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(20, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(21, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg, outdir="../data/valid/")
# 	gen_ext_feature(23, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(24, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(25, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(26, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(27, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(29, feat_names, label_names, is_aggregated, is_normalization, down_rate, 'down')
# 	gen_ext_feature(30, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(31, feat_names, label_names, is_aggregated, is_normalization, down_rate, 'right')
# 	gen_ext_feature(32, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg, outdir="../data/train/Temp/")
# 	gen_ext_feature(33, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg)
# 	gen_ext_feature(34, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.sil_flg)
# 	for i in range(15, 17):
# 		gen_ext_feature(i, feat_names, label_names, is_aggregated, is_normalization, down_rate, cfg.unk_flg, outlier_path='../data/outlier/orig/')
		
		
if __name__ == "__main__":
	test()
