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
sys.path.append('../../')
from base import data_util

from config import Config as cfg
import xgboost as xgb
import os
import re
from num2words import num2words
import inflect
from tables.tests.create_backcompat_indexes import nrows
import fst
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from pandas.tseries.offsets import _is_normalized
import datetime
from scipy import sparse
import random
import cPickle as pickle
from bokeh.server.protocol.messages import index

def standardization(X):
	x_t = preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=True)
	return x_t


def call_feature_func(data, feats_name, is_normalized=False):
# 	feats = pd.DataFrame()
	feats = {}
	for name in feats_name:
		func_name = "extract_{feat}_feature".format(feat = name)
		ret = eval(func_name)(data, is_normalized)
# 		if type(ret)==list:
# 			for t in ret:
# 				feats[t[0]] = t[1]
# 		else:
		feats[name] = ret
		
	return feats





def extract_sentence_id_feature(data, is_normalized=False):
	def q_val(val):
		return val
	vals = data.apply(lambda x: q_val(x['sentence_id']), axis=1, raw=True)
	if is_normalized:
		vals = standardization(vals)
	return vals



def extract_token_id_feature(data, is_normalized=False):
	def q_val(val):
		return val
	vals = data.apply(lambda x: q_val(x['token_id']), axis=1, raw=True)
	if is_normalized:
		vals = standardization(vals)
	return vals



def extract_class_feature(data, is_normalized=False):
	fac = pd.factorize(data['class'])
	return fac[0]

def extract_before_len_feature(data, is_normalized=False):
	def q_val(val):
		return len(str(val))
	vals = data.apply(lambda x: q_val(x['before']), axis=1, raw=True)
	if is_normalized:
		vals = standardization(vals)
	return vals

# def extract_before_feature(data, is_normalized=False):
# 	
# 	def context_window_transform(data, pad_size):
# 		pre = np.zeros(cfg.max_num_features)
# 		pre = [pre for x in np.arange(pad_size)]
# 		data = pre + data + pre
# 		neo_data = []
# 		for i in np.arange(len(data) - pad_size * 2):
# 			row = []
# 			for x in data[i : i + pad_size * 2 + 1]:
# 				row.append([cfg.boundary_letter])
# 				row.append(x)
# 			row.append([cfg.boundary_letter])
# 			neo_data.append([int(x) for y in row for x in y])
# 		return neo_data
	
# 	def q_val(val):
# 		x_row = np.ones(cfg.max_num_features, dtype=int) * cfg.space_letter
# 		for xi, i in zip(list(str(val)), np.arange(cfg.max_num_features)):
# 			x_row[i] = ord(xi)
# 		return x_row
# 	
# 	
# 	x_data = []
# 	for x in data['before']:
# 		x_data.append(q_val(x))
# 		
# 	x_data = context_window_transform(x_data, cfg.pad_size)
# 	
# 		
# 	x_data = zip(*x_data)
# 	
# 	
# 	
# 	
# 	n = len(x_data)
# 	
# 	vals = []
# 	
# 	for r, i in zip(x_data, range(n)):
# 		name = "before_%d"%(i)
# 
# 		vals.append((name, pd.Series(r)))
# 
# 	return vals



def extract_after_feature(data, is_normalized=False):
	def q_val(val):
		return val
	vals = data.apply(lambda x: q_val(x['after']), axis=1, raw=True)
	if is_normalized:
		vals = standardization(vals)
	return vals


def extract_2gram_feature(data, is_normalized=False):
	return extract_ngram_feature(data, 2)

def down_sampling_discard(word, before_after_dict,rate=0.025):
	
	val = random.uniform(0,1)
	afters = before_after_dict.get(word, None)
	if word in cfg.dic_constant and val > rate and (afters is not None and word in afters):
		return True
	else:
		return False

def extract_ngram_feature(data, ngram=2):
	ngram_features = []
	discard_index_set = set()
	before_after_dict = {}
	N = len(data)
	def add_to_dic(key, val):
		if before_after_dict.has_key(key):
			vals = before_after_dict[key]
			if val not in vals:
				vals.append(val)
		else:
			vals = []
			vals.append(val)
			before_after_dict[key] = vals
	
	for i in range(N):
		
		if i % 100000 == 0:
			print i
		
	
		skip_flg = down_sampling_discard(data.at[i, 'before'], before_after_dict, 0.0000)
		if skip_flg:
			discard_index_set.add(i)
			continue
		
		add_to_dic(data.at[i, 'before'], data.at[i, 'after'])
		
		token_id = data.at[i, 'token_id']
# 		token_id = row['token_id'].values[0]
		MaxToken = token_id
		k = i + 1
		while k < N:
			token_id_c = data.at[k, 'token_id']
			if token_id_c > token_id:
				MaxToken = token_id_c
				k = k + 1
			else:
				break
		
		context = []
		for t, m in zip(range(token_id - ngram, token_id + ngram + 1), range(i - ngram, i + ngram + 1)):
# 			print "i:%d,m:%d,t:%d"%(i, m, t)
# 			char_arr = [0]*cfg.word_vec_len
			char_list = []
			
			if t < 0:
# 				char_list = cfg.start_word
				context.append(cfg.boundary_start)
# 				print "i:%d,m:%d,t:%d, start"%(i, m, t)
			elif t > MaxToken:
# 				print "i:%d,m:%d,t:%d, end"%(i, m, t)
# 				char_list = cfg.end_word
				context.append(cfg.boundary_end)
			else:
# 				print "i:%d,m:%d,t:%d, mid"%(i, m, t)
				char_list = [ord(v) for v in list(str(data.at[m, 'before']))]
				
				if t == token_id:
					context.append(cfg.input_split)
					context.extend(char_list)
					context.append(cfg.input_split)
				elif t == (token_id - 1):
# 					context.append(cfg.space_letter)
					context.extend(char_list)
# 					context.append(cfg.space_letter)
				else:
					context.extend(char_list)
					context.append(cfg.space_letter)
# 				for l in range(0, len(char_list)):
# 					char_arr[l + 1] = char_list[l]
			#insert split flag in both sides of the input word

# 		print context
		ngram_features.append(context)
# 	features = ngram_features
	features = sequence.pad_sequences(ngram_features, maxlen=cfg.max_input_len, padding='post')
	print "complated extract ngram feature!"
	return features, discard_index_set


def extract_all_features(df_list, is_normalized=False):
	data_all = pd.concat(df_list)
	feats_name = [
# 				'sentence_id',
# 				'token_id',
# 				'class',
				'2gram',
# 				'after',
				]

	feats = call_feature_func(data_all, feats_name, is_normalized)
	return feats

def load_dict(file):
	df = pd.read_csv(file)
	dic = {}
	print "real output word num:%d"%len(df)
	for i in range(len(df)):
		key = df.at[i, 'word']
		dic[key] = i + 4
	
	dic[cfg.start_flg] = cfg.start_flg_index
	dic[cfg.end_flg] = cfg.end_flg_index
	dic[cfg.copy_flg] = cfg.copy_flag_index
	dic[cfg.non_flg] = cfg.non_flg_index
	return dic


def extract_y_info(data, discard_index_set=None):

	dic = load_dict('../data/out_vocab.csv')

	
	def covert(x):
		words = str(x).strip().split(' ')
		values = map(lambda x: dic.get(x, cfg.copy_flag_index), words)
		values.insert(0, dic[cfg.start_flg])
		values.append(dic[cfg.end_flg])
		return values
	list_values = []
	for i in range(len(data)):
		if discard_index_set is not None and i in discard_index_set:
			continue
		values = covert(data.at[i, 'after'])
		list_values.append(values)
# 	y = data['after'].apply(lambda x : covert(x))
	
# 	y = list_values
	y = sequence.pad_sequences(list_values, maxlen=cfg.max_output_len, padding='post')
	print "complated extract y info!"
	return y

def extract_y_class_info(data):
	y = None
	if 'class' in data.columns:
		y = call_feature_func(data, ['class'])
	return y

def dump(obj, path):
	f = open(path, 'wb')
	pickle.dump(obj, f)
	f.close()
def load(path):	
	f = open(path, 'rb')
	obj = pickle.load(f)
	f.close()
	return obj

def gen_features(df, is_normalized=False):
	gram, discard_index_set = extract_2gram_feature(df, is_normalized)
	y = extract_y_info(df, discard_index_set)
# 	if y is not None:
# 		y = to_categorical(y.tolist())
		
	# truncate and pad input sequences
# 	X = sequence.pad_sequences(x.values, maxlen=cfg.max_input_len)
	
	dump(discard_index_set, '../data/discard_index_set')
	
	df['pos'] = pd.Series(range(len(df)))
	df_filted = df[~df['pos'].isin(list(discard_index_set))]
	del df['pos']
	del df_filted['pos']
	df_filted['id'] = df.apply(lambda x: str(x['sentence_id']) + "_" + str(x['token_id']), axis=1)
	print 'remained token num:%d, discard num %d'%(len(df_filted), len(discard_index_set))
	display_feature_info(df_filted, 'class')
	df_filted.to_csv('../data/en_train_filted.csv', index=False)
	return gram, y
	
def down_sampling(x, y):
	del_index = []
	for i, y_sub in enumerate(y):
		if y_sub[0] == (cfg.start_flg_index) and (y_sub[1] == cfg.unk_flg_index) and (y_sub[2]==cfg.end_flg_index):
			val = random.uniform(0,1)
			if val > 0.05:
				del_index.append(i)
	x = np.delete(x, del_index, axis=0)
	y = np.delete(y, del_index, axis=0)
	return x, y

def gen_data_from_npz(file_npz, range=None):
	x_t, y_t = load_numpy_datas(file_npz)
	
	df = pd.read_csv('../data/en_train_filted.csv')
# 	df['id'] = df.apply(lambda x: str(x['sentence_id']) + "_" + str(x['token_id']), axis=1)
	
	
	if range is not None:
		x_t = x_t[range[0]:range[1]]
		y_t = y_t[range[0]:range[1]]
	
	df_train, df_valid, arr_train, arr_valid = train_test_split(df, np.hstack([x_t, y_t]),
                                                      test_size=10000, random_state=0)
	
	x_train = arr_train[:, 0:x_t.shape[1]]
	y_train = arr_train[:, x_t.shape[1]:]
	
	x_valid = arr_valid[:, 0:x_t.shape[1]]
	y_valid = arr_valid[:, x_t.shape[1]:]
# 	exception_ind = np.where(y_t==1)
# 	print exception_ind
# 	df = pd.read_csv('../data/en_train_filted_small.csv')
# 	df['id'] = df.apply(lambda x: x['sentence_id'] + "_" + x['token_id'], axis=1)
# 	all_index = df.index.tolist()
# 	
# 	
# 	vaild_index = random.sample(range(0, len(df)), int(0.1 * len(df)));
# 	train_index = list(set(all_index) ^ set(vaild_index))
# 	
# 	df_train = df.iloc[train_index]
# 	df_vaild = df.iloc[vaild_index]

# 	x_train = x_t
# 	y_train = y_t

# 	x_valid = x_train[0:1000]
# 	y_valid = y_train[0:1000]
	print "Load Train data:%d, Vaild data:%d"%(len(x_train), len(x_valid))
	return x_train, y_train, x_valid, y_valid, df_train, df_valid

# def down_sampling(x_train, y_train):
# 	
# 	for i, y in enumerate(y_train):

def load_input_dict():
	df = pd.read_csv('../data/constant_vocab.csv')
	in_vocab = set()
	df['word'].apply(lambda x: in_vocab.add(x))
	return in_vocab

def gen_input_plain_vocab():
	df = pd.read_csv('../data/en_train.csv')
	df_plain = df[df['before']==df['after']]
	voc = df_plain['before'].value_counts().sort_values()
	
# 	df_out = pd.DataFrame(voc, columns=['word', 'count'])
	voc.to_csv('../data/in_plain_vocab.csv', index=True)
	print "save in-plain-vocab with size:%d"%(len(voc))

def gen_data(df_train, df_test=None, is_resample=False, is_normalized=False):
	
	x_train, y_train = gen_features(df_train, is_normalized)
	
	x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                      test_size=0.1, random_state=2017)
	
	x_test = None
	y_test = None
	if df_test is not None:
		x_test, y_test = gen_features(df_test, is_normalized)
	
	
	x_t = x_train
	y_t = y_train
	if is_resample:
		x_t, y_t = data_util.resample(x_train, y_train)
	return x_t, y_t, x_valid, y_valid, x_test, y_test

def num_to_words(str, group=3):
	
	p = inflect.engine()
	words = p.number_to_words(str, group=group, andword='', zero='o')
	words = words.replace(', ', ' ')
	words = words.replace('-', ' ')
	print words
	return words

def split_ADDRESS(str_before):
	num = re.findall(r'\d+', str_before)[0]
	index = str_before.find(num)
	prefix = str_before[0:index].strip()
	
# 	print 'Found prefix \'%s\' and number \'%s\' in %d of \'%s\''%(prefix, num, index, str_before)
	return prefix, num
	


	
def convert_ADDRESS(str_before):
# 	str_before = str_before.replace('.',' ')
# 	str_before = str_before.replace('-',' ')
	str_before = str_before.replace(',',' ')
	
	prefix, num = split_ADDRESS(str_before)
	
# 	if len(prefix) 


def group_data_address():
	path = '../data/grouped/'
	df = pd.read_csv('../data/grouped/ADDRESS.csv')
	
	
	
	df['prefix'] = df['before'].apply(lambda x : split_ADDRESS(x)[0])
	df['number'] = df['before'].apply(lambda x : split_ADDRESS(x)[1])
	

	grouped = df.groupby(['prefix'])
	grouped
	out = pd.DataFrame()
	groups = []
	for name, group in grouped:
		group['len'] = group['before'].apply(lambda x : len(x))
		sg = group.sort_values(['len'])
# 		print sg
		groups.append(sg)
# 	grouped.sort('len')
	out = pd.concat(groups)
# 	out.drop('prefix',axis=1, inplace=True)
	del out['prefix']
	del out['number']
	del out['len']
	out.to_csv('../data/grouped/ADDRESS_grouped.csv')
def group_data():
	
	path = '../data/groupedtest/'
	df = pd.read_csv('../data/en_test.csv')
	grouped = df.groupby(['class'])
	
	for name, group in grouped:
		if name == 'ADDRESS':
			group['len'] = group['before'].apply(lambda x : len(x))
			group.sort_values(['len'], inplace=True)
			group.drop('len',axis=1, inplace=True)
		
		group.to_csv(path + name + '.csv', index=False)	
		print 'Saved data of %s'%(name)
		
# 		df_info = pd.DataFrame()
# 		df_info['before'] = grouped['before']
# 		df_info['after'] = grouped['after'] 
# 		df_info.to_csv('')

def display_feature_info(df, name):
	cnts = df[name].value_counts().sort_values()
	num_class = len(cnts)
	print "num_class of %s is:%d"%(name, num_class)
	print cnts

def display_sentence(id):
	df_train = pd.read_csv('../data/en_train.csv')
	df = df_train[df_train['sentence_id']==id]
	print df

def test():
	df_train = pd.read_csv('../data/en_train.csv')
	out_path = '../data/'
	
	x_train = call_feature_func(df_train, ['before']).values
	print len(x_train)
# 	display_feature_info(feats, 'before_0')
	y_train = extract_y_info(df_train).values
	
	x_train, x_valid, y_train, y_valid= train_test_split(x_train, y_train,
                                                      test_size=0.1, random_state=2017)
	
	num_class = 16
	dtrain = xgb.DMatrix(x_train, label=y_train)
	dvalid = xgb.DMatrix(x_valid, label=y_valid)
	watchlist = [(dvalid, 'valid'), (dtrain, 'train')]
	
	param = {'objective':'multi:softmax',
	         'eta':'0.3', 'max_depth':10,
	         'silent':1, 'nthread':-1,
	         'num_class':num_class,
	         'eval_metric':'merror'}
	model = xgb.train(param, dtrain, 50, watchlist, early_stopping_rounds=20,
	                  verbose_eval=10)
	
	
	labels = [u'PLAIN', u'PUNCT', u'DATE', u'LETTERS', u'CARDINAL', u'VERBATIM',
       u'DECIMAL', u'MEASURE', u'MONEY', u'ORDINAL', u'TIME', u'ELECTRONIC',
       u'DIGIT', u'FRACTION', u'TELEPHONE', u'ADDRESS']
	pred = model.predict(dvalid)
	pred = [labels[int(x)] for x in pred]
	y_valid = [labels[x[0]] for x in y_valid]
	x_valid = [ [ chr(x) for x in y[2 + cfg.max_num_features: 2 + cfg.max_num_features * 2]] for y in x_valid]
	x_valid = [''.join(x) for x in x_valid]
	x_valid = [re.sub('a+$', '', x) for x in x_valid]
	
	df_pred = pd.DataFrame(columns=['data', 'predict', 'target'])
	df_pred['data'] = x_valid
	df_pred['predict'] = pred
	df_pred['target'] = y_valid
	df_pred.to_csv(os.path.join(out_path, 'pred.csv'))
	
	df_erros = df_pred.loc[df_pred['predict'] != df_pred['target']]
	df_erros.to_csv(os.path.join(out_path, 'errors.csv'), index=False)
	
	model.save_model(os.path.join(out_path, 'xgb_model'))
	
# 	display_feature_info(df_train, 'sentence_id')
# 	display_feature_info(df_train, 'token_id')
# 	display_feature_info(df_train, 'after')

def display_varible_terms():
	df_train = pd.read_csv('../data/en_train.csv')
	df1 = df_train[(df_train['class']=='PLAIN') & (df_train['before'] != df_train['after'])]
	df2 = df_train[(df_train['class']=='PLAIN') & (df_train['before'] == df_train['after'])]
	dict1 = df1['before'].value_counts().to_dict()
	dict2 = df2['before'].value_counts().to_dict()
	
	inter = dict.fromkeys([x for x in dict1 if x in dict2])
	print len(inter)
	
	df_list = []
	for k, v in inter.iteritems():
		df_list.append(df2[df2['before'] == k])
	df_out = pd.concat(df_list)
	df_out.to_csv('../data/PLAIN_var.csv', index=False)
	
	

def gen_spec_dict(class_name):
	df_train = pd.read_csv('../data/en_train.csv')
	df = df_train[(df_train['class']==class_name) & (df_train['before'] != df_train['after'])]
	grouped = df.groupby(['before'])
	
# 	groups = []
	spec_list = []
	for name, group in grouped:
		cnts = group['after'].value_counts().to_dict()
		for k, v in cnts.iteritems():
			spec_list.append([name, k])
		
# 		groups.append(group)
	df_spec = pd.DataFrame(spec_list, columns=['before', 'after'])
	print spec_list
# 	out = pd.concat(groups)
	df_spec.to_csv('../data/{name}_spec.csv'.format(name=class_name), index=False)
# 	out.to_csv('../data/PLAIN_discrepancy.csv', index=False)

def gen_in_vocab():
	df_data = pd.read_csv('../data/en_train.csv')
	words = []
	df_data['before'].apply(lambda x: words.append(x))
	voc = Counter(words).most_common()
	df_out = pd.DataFrame(voc, columns=['word', 'count'])
	df_out.to_csv('../data/in_vocab.csv', index=False)
	print "save in-vocab with size:%d"%(len(df_out))
	
def gen_in_char_vocab():
	df_data = pd.read_csv('../data/en_train.csv')
	chars = []
	#store the unicode of each char in the terms
	df_data['before'].apply(lambda x: chars.extend([ord(c) for c in list(str(x))]))
	voc = Counter(chars).most_common()
	df_out = pd.DataFrame(voc, columns=['char', 'count'])
	df_out.to_csv('../data/in_char_vocab.csv', index=False)
	print "save in-char-vocab with size:%d"%(len(df_out))
	
def gen_out_vocab():
	df_data = pd.read_csv('../data/en_train.csv')
	df = df_data[df_data['before'] != df_data['after']]
	words = []
	df['after'].apply(lambda x: words.extend(str(x).split(' ')))
	voc = Counter(words).most_common()
	df_out = pd.DataFrame(voc, columns=['word', 'count'])
	df_out.to_csv('../data/out_vocab.csv', index=False)
	print "save out-vocab with size:%d"%(len(df_out))

		
def display_longest_sentence():
	df_train = pd.read_csv('../data/en_train.csv')
# 	print "word number:"
# 	df_train['token_id'].value_counts()
	
	
	grouped = df_train.groupby(['sentence_id'])
	
	sentences = []
	lengths = []
	print "Total sentence's number:%d"%(len(grouped))
	for id, group in grouped:
		sentences.append(id)
		lens = []
		group['before'].apply(lambda x : lens.append(len(str(x))))
		lengths.append(sum(lens))
# 		group.sort_values(['len'], inplace=True)
# 		group.drop('len',axis=1, inplace=True)
	print max(lengths)
# 	df_out = pd.DataFrame(sentences, lengths, columns=['sentence_id', 'char_num'])
# 	df_train['char_num'].value_counts()
	
	
def display_max_ngram_feature_length(data, n=2):
 	feat = extract_ngram_feature(data, n)
 	lens = map(lambda x: len(x), feat)
 	print "max %d gram feature length is %d"%(n, max(lens))
 	
def display_max_y_length(data):
	y_train = extract_y_info(data)
	lens = map(lambda x: len(x), y_train)
	print "max y length is %d"%(max(lens))

def save_numpy_data(file, np_data):
	f = file(file, "wb")
	np.save(f, np_data)

def load_numpy_data(file):
	f = file(file, "rb")
	return np.load(f)

def load_numpy_datas(file):
	r = np.load(file)
	return r['x_t'], r['y_t']

def filter_data_by_len(data):
	data['len'] = data['before'].apply(lambda x:len(str(x)))
	print data['len'].value_counts().sort_values()
	
	data_large = data[data['len'] > cfg.max_token_char_num]
	inlist = data_large['sentence_id'].tolist()
	
	filted = data[(data['len'] <= cfg.max_token_char_num) & (~data['sentence_id'].isin(inlist))]
	
	del filted['len']
	return filted

def gen_constant_dict():
	data = pd.read_csv('../data/en_train.csv')
	before_after_dict = {}
	def add_to_dic(key, val):
		if before_after_dict.has_key(key):
			vals = before_after_dict[key]
			if val not in vals:
				vals.append(val)
		else:
			vals = []
			vals.append(val)
			before_after_dict[key] = vals
	data.apply(lambda x: add_to_dic(x['before'], x['after']), axis=1)
	list_words = []
# 	list_after = []
	
	for key, value in before_after_dict.items():
		if len(value) == 1 and key == value[0]:
			list_words.append(key)
	
	df_out = pd.DataFrame(list_words, columns=['word'])
	df_out.to_csv('../data/constant_vocab.csv', index=False)
	print "save constant-vocab with size:%d"%(len(df_out))


def display_data_info(data):
	display_max_ngram_feature_length(data)
	display_max_y_length(data)
	
	cnt = extract_before_len_feature(data).value_counts().sort_values()
# 	print cnt



if __name__ == "__main__":
	
	df_train = pd.read_csv('../data/en_train_filted.csv')
# 	df_test = pd.read_csv('../data/en_test_filted.csv')
# 	Len_orgin = len(df_train)
# 	print 'length %d'%(Len_orgin)
# 	data = filter_data_by_len(df_train)
# 	Len_filted = len(data)
# 	data.to_csv('../data/en_test_filted.csv', index=False)
# 	print 'reduce length %d -> %d'%(Len_orgin, Len_filted)
# 	df_test = pd.read_csv('../data/en_test.csv')
# 	display_data_info(df_train)
# 	display_max_ngram_feature_length(df_test)
	
	begin = datetime.datetime.now()

	x_t, y_t = gen_features(df_train)
	
	print len(x_t)
	print len(y_t)
# 	print len(x_valid)
# 	print len(y_valid)
	end = datetime.datetime.now()
	print 'run time:' + str(end - begin)
# 	
# 	
	begin = datetime.datetime.now()
# 	coo_x_t = sparse.coo_matrix(x_t)
# 	coo_y_t = sparse.coo_matrix(y_t)
# 	coo_x_valid = sparse.coo_matrix(x_valid)
# 	coo_y_valid = sparse.coo_matrix(y_valid)
# 	sparse.mmwrite('coo_x_t.mtx', coo_x_t)
	np.savez("../data/train.npz", x_t = x_t, y_t = y_t)
# 	x_train, y_train, x_valid, y_valid = gen_data_from_npz("../data/train.npz")
# 	
# 	print len(x_train)
# 	print len(y_train)
# 	print len(x_valid)
# 	print len(y_valid)
# 	end = datetime.datetime.now()
# 	print 'run time:' + str(end - begin)
#   	print y_train
# 	gen_output_vocab()
#  	display_varible_terms()
# 	group_data_address()
# 	num_to_words('105.6',0)
# 	print num2words(105.6)
# 	display_sentence(78354)
# 	group_data()
# 	display_longest_sentence()

