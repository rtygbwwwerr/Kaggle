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

sys.path.append('../../')
from base import data_util
from config import Config as cfg
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from python_speech_features import mfcc, logfbank, fbank, delta
from keras.preprocessing import sequence
cfg.init()



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


	
	
def logspecgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
	nperseg = int(round(window_size * sample_rate / 1e3))
	noverlap = int(round(step_size * sample_rate / 1e3))
	freqs, times, spec = signal.spectrogram(audio,
	                                fs=sample_rate,
	                                window='hann',
	                                nperseg=nperseg,
	                                noverlap=noverlap,
	                                detrend=False)
	return np.log(spec.T.astype(np.float32) + eps)

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
			
			sample = (label, entry)
			if uid in valset:
				val.append(sample)
			else:
				train.append(sample)
	
	print('There are {} train and {} val samples'.format(len(train), len(val)))
	return train, val

def pad_audio(samples, max_len=cfg.sampling_rate):
	if len(samples) >= max_len: return samples
	else: return np.pad(samples, pad_width=(max_len - len(samples), 0), mode='constant', constant_values=(0, 0))

def gen_silence_data(resampline_sil_rate=200, output="../data/train/ext0/"):
	inputs = gen_input_paths("../data/train/audio/_background_noise_/", ".wav")
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

def get_wav(data, is_normalization=True):
	x = []
	y = []
	y_c = []
	y_w = []
	for (label, fname) in data:
# 		print fname
		_, wav = wavfile.read(fname)
		wav = pad_audio(wav)
# 		wavfile.write("../data/test_original.wav", cfg.sampling_rate, wav)
		wav = wav.astype(np.float32) / np.iinfo(np.int16).max
		
# 		# be aware, some files are shorter than 1 sec!
# 		if len(wav) < cfg.sampling_rate:
# 			continue
		if is_normalization:
			wav = standardization(wav)

		x.append(wav)

		y.append([np.int32(cfg.n2i(label))])
		y_c.append(get_char_indies(label))
		y_w.append(get_word_indies(label))
# 	y = to_categorical(y, num_classes = cfg.CLS_NUM)
	return x, y, y_c, y_w

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

def mfcc10(x, sampling_rate):
	return mfcc(x, sampling_rate, nfft=1024, winlen=0.04, winstep=0.02, nfilt=49, numcep=10)

def mfcc40(x, sampling_rate):
	return mfcc(x, sampling_rate, nfft=1024, winlen=0.04, winstep=0.02, nfilt=49, numcep=40)

def logfbank40(x, sampling_rate):
	return logfbank(x, sampling_rate, nfilt=40, lowfreq=300, highfreq=3000)
def logfbank80(x, sampling_rate):
	return logfbank(x, sampling_rate, nfilt=80, lowfreq=300, highfreq=3000)

def rawwav(audio, sample_rate):
	return audio

def get_features(data, name, sampling_rate, **arg):
	
	print "call {}() func".format(name)
	output = map(lambda x : eval(name)(x, sampling_rate), data)
# 	output = np.vstack(output)
# 	print output[0]
	arr = np.array(output)
	print arr.shape

	return arr


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



def gen_train_feature(name, is_normalization=True, down_rate=1.0, data_dir="../data/"):
	train, val = load_data(data_dir)
	sampling_rate = int(cfg.sampling_rate * down_rate)
	x, y, y_c, y_w = get_wav(train, is_normalization=is_normalization)
	x_wav = np.array(x)
	
	if down_rate < 1.0:
		x = down_sample(x, down_rate)
# 	x = pad_data(x)
	
	x = get_features(x, name, sampling_rate)
	
	v_x, v_y, v_y_c, v_y_w = get_wav(val)
	v_x_wav = np.array(v_x)
# 	v_x = pad_data(v_x)
	if down_rate < 1.0:
		v_x = down_sample(v_x, down_rate)
	v_x = get_features(v_x, name, sampling_rate)
	
	
	np.savez(make_file_name("../data/train/train", "npz", sampling_rate, name), x_wav=x_wav,x = x, y = y, y_c = y_c, y_w = y_w)
	np.savez(make_file_name("../data/valid/valid", "npz", sampling_rate, name), x_wav=v_x_wav,x = v_x, y = v_y, y_c = v_y_c, y_w = v_y_w)
	print "sampling rate:{}".format(sampling_rate)
	print "feature shape:{}*{}".format(x[0].shape[0], x[0].shape[1])
	print "completed gen training data..."
	
def gen_test_feature(name, is_normalization=True, down_rate=1.0, data_dir="../data/test/"):
	paths = gen_input_paths(data_dir + "audio/", ".wav")
	print "test data num:{}".format(len(paths))
	x = []
	names = []
	sampling_rate = int(cfg.sampling_rate * down_rate)
	for i,fname in enumerate(paths):
		_, wav = wavfile.read(fname)
		wav = pad_audio(wav)
		wav = wav.astype(np.float32) / np.iinfo(np.int16).max
		if is_normalization:
			wav = standardization(wav)
		L = cfg.sampling_rate

		if len(wav) > L:
			beg = np.random.randint(0, len(wav) - L)
		else:
			beg = 0
		wav = wav[beg: beg + L]
		names.append(os.path.basename(fname))
		x.append(wav)
		
		if i % 10000 == 0:
			print "read {} files".format(i)
	x_wav = np.array(x)
	print "completed read wav files, start extracting fearures..."
	if down_rate < 1.0:
		x = down_sample(x, down_rate)
	x = get_features(x, name, sampling_rate)
	
	np.savez(make_file_name(data_dir + "test", "npz", sampling_rate, name), x = x, x_wav=x_wav, names=names)
	print "sampling rate:{}".format(sampling_rate)
	print "feature shape:{}*{}".format(x[0].shape[0], x[0].shape[1])
	print "completed gen test data..."
	
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

def gen_input_paths(root_path="../data/ext/", file_ext_name=".csv"):
	list_path = os.listdir(root_path)
	
# 	total_num = 0
	paths = []
	for path in list_path:
		file_path = os.path.join(root_path, path)
		if os.path.isfile(file_path) and file_path.endswith(file_ext_name):
			paths.append(file_path)
# 	paths.append('../data/en_train.csv')
	
	return paths
def get_file_trim(type="npz", filter_trim=None):
	return ".{}".format(type) if (filter_trim == "") or (filter_trim is None) else "_{}.{}".format(filter_trim, type)

def make_file_trim(type="npz", *args):
	name = ".{}".format(type)
	if args is not None and len(args) > 0:
		for arg in args[0]:
			name = "_" + str(arg) + name
	
	return name
	
	
def get_data_from_files(root_path, data_names, filter_trim=None, input_num=0, pad_id=0):
	trim = get_file_trim(filter_trim = filter_trim)
	print "process file type:" + trim
	paths = gen_input_paths(root_path, trim)

	data_dict = {}
	print "We'll load {} files...".format(len(paths))
	for path in paths:
		data = np.load(path)
		print "load {} data from:{}".format(len(data[data_names[0]]), path)
		for name in data_names:
			item = data[name]
			if name in data_dict:
				data_dict[name].append(item)
			else:
				data_dict[name] = [item]
			print item.shape
		
	for name in data_names:
		data_dict[name] = concate_datas(data_dict[name], pad_id)
	
	ret = None
	if input_num > 0:
		ret = (data_dict[name][:input_num] for name in data_names)
	else:
		ret = (data_dict[name] for name in data_names)
		
	return ret




def concate_datas(data_list, pad_id=0):
	datas = []
	if check_type_consistency(data_list):
		datas = np.vstack(data_list)
	else:
		for data in data_list:
			datas.extend(list(data))
		if pad_id != None:
			datas = sequence.pad_sequences(datas, dtype=np.int32, padding='post', value=pad_id)
	return datas

def check_type_consistency(data_list):
	for data in data_list:
		if not data_util.isDigitArrType(data):
			return False
	return True

def load_data_ext(data_dir, default_label):
	paths = gen_input_paths(data_dir, ".wav")
	samples = []
	for path in paths:
		index = path.find('-')
		label = default_label
		if index > -1:
			label = path[index + 1:-4]
			if label == 'silence':
				label = cfg.sil_flg
			if label == 'unknown':
				label = cfg.unk_flg
		samples.append((label, path))
		
	return samples		

def gen_ext_feature(id, name, is_normalization, down_rate=1.0, default_label=cfg.sil_flg):
	data_dir = "../data/train/ext{}/".format(id)
# 	output = "../data/train/train_ext{}_{}.npz".format(id, name)
	
	sampling_rate = int(cfg.sampling_rate * down_rate)
	output = make_file_name("../data/train/train_ext{}".format(id), "npz", sampling_rate, name)
	samples = load_data_ext(data_dir, default_label)			
	x, y, y_c, y_w = get_wav(samples, is_normalization=is_normalization)
	x_wav = np.array(x)
	if down_rate < 1.0:
		x = down_sample(x, down_rate)
	x = get_features(x, name, sampling_rate)
	
	np.savez(output, x = x, x_wav = x_wav, y = y, y_c = y_c, y_w = y_w)
	print "sampling rate:{}".format(sampling_rate)
	print "feature shape:{}*{}".format(x[0].shape[0], x[0].shape[1])
	print "completed gen ext \"{}\" feature of {} files from ext{}.".format(default_label, x.shape[0], id)

def init_noise_array(paths):
	
	data = []
	for path in paths:
		wpaths = gen_input_paths(path, ".wav")
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

noise_array = init_noise_array(['../data/train/audio/_background_noise_',"../data/train/ext1/", "../data/train/ext2/"])
	


def mix_noise(data, rate=0.005):
# 	L = data.shape[-1]
	axis = len(data.shape) - 1
	num = 1 if axis <= 0 else data.shape[0]
	noises = None
	if num > 1:
		noises = np.vstack(random.sample(noise_array, num))
	else:
		noises = random.sample(noise_array, num)[0]
	return data + noises * rate
	
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
	data = librosa.effects.time_stretch(data, rate)
	if len(data)>input_length:
		data = data[:input_length]
	else:
		data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
	
	return data

def down_sample(data, rate=0.5):
	return map(lambda x : signal.resample(x, int(cfg.sampling_rate * rate)), data)

def random_call(data, ops):
	
	out = data
	for op in ops:
		name = op[0]
		
		val = random.uniform(0, 1)
		if val < op[1]:
			rate = random.uniform(op[2][0], op[2][1])
			out = eval(name)(out, rate)
	return out

def augmentation(data, outdir='../data/train/ext8/', 
				ops=[("mix_noise", 0.85, (1.0, 3.0)), ("shift", 0.8, (-0.15, 0.15)), ("stretch", 0.75, (0.75, 1.35))]):
	
	cnt = 1
	for (label_id, fname) in data:
# 		print fname
		_, wav = wavfile.read(fname)
		if len(wav) > cfg.sampling_rate:
			print "skip overlength file {}".format(fname)
			continue
		wav = pad_audio(wav)
		wav = wav.astype(np.float64)
		wav = random_call(wav, ops)
		fname = os.path.basename(fname)
		index = fname.find(".wav")
		if "-" not in fname:
			fname_out = fname[0:index] + "-" + cfg.i2n(label_id) + ".wav"
			fname_out = outdir + fname_out
			wav = wav.astype(np.int16)
			
			id = 1
			while os.path.exists(fname_out):
				fname_out = fname[0:index] + "_" + str(id) +"-" + cfg.i2n(label_id) + ".wav"
				fname_out = outdir + fname_out
				id += 1
				print id
				
			wavfile.write(fname_out, cfg.sampling_rate, wav)
			print "processed num:{}".format(cnt)
			cnt += 1
			
			
# def remove_silence():
# 	v = VoiceActivityDetector(filename)
	

	
	
def gen_augmentation_data():
	train, _ = load_data()
	augmentation(train)
	
def test():
	#name should in ["mfcc", 'mfcc10', 'mfcc40',"logfbank", 'logfbank40', 'logfbank80', "logspecgram", 'rawwav']
# 	func = logfbank
# 	print make_file_name("../data/train/train", "npz", 16000, "logspecgram")
	name = "logfbank"
	is_normalization = True
	down_rate=0.5
# 	gen_augmentation_data()
# 	default_label = 'off'
 	gen_train_feature(name, is_normalization, down_rate)
	gen_test_feature(name, is_normalization, down_rate)
 	gen_ext_feature(0, name, is_normalization, down_rate, cfg.sil_flg)
 	gen_ext_feature(1, name, is_normalization, down_rate, cfg.sil_flg)
 	gen_ext_feature(2, name, is_normalization, down_rate, cfg.sil_flg)
 	gen_ext_feature(3, name, is_normalization, down_rate, 'off')
 	gen_ext_feature(4, name, is_normalization, down_rate, 'no')
 	gen_ext_feature(5, name, is_normalization, down_rate, 'up')
 	gen_ext_feature(6, name, is_normalization, down_rate, 'stop')
 	gen_ext_feature(7, name, is_normalization, down_rate, 'unknown')
 	gen_ext_feature(8, name, is_normalization, down_rate, 'unknown')
if __name__ == "__main__":
	test()
