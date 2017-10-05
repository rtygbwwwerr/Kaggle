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
# sys.path.append('../../')
from base import data_util

from config import Config as cfg
import xgboost as xgb
import os
import re
from num2words import num2words
import inflect
from tables.tests.create_backcompat_indexes import nrows
import fst

def standardization(X):
	x_t = preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=True)
	return x_t


def call_feature_func(data, feats_name, is_normalized=False):
	feats = pd.DataFrame()
	for name in feats_name:
		func_name = "extract_{feat}_feature".format(feat = name)
		ret = eval(func_name)(data, is_normalized)
		if type(ret)==list:
			for t in ret:
				feats[t[0]] = t[1]
		else:
			feats[name] = eval(func_name)(data, is_normalized)
		
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

def extract_before_feature(data, is_normalized=False):
	
	def context_window_transform(data, pad_size):
		pre = np.zeros(cfg.max_num_features)
		pre = [pre for x in np.arange(pad_size)]
		data = pre + data + pre
		neo_data = []
		for i in np.arange(len(data) - pad_size * 2):
			row = []
			for x in data[i : i + pad_size * 2 + 1]:
				row.append([cfg.boundary_letter])
				row.append(x)
			row.append([cfg.boundary_letter])
			neo_data.append([int(x) for y in row for x in y])
		return neo_data
	
	def q_val(val):
		x_row = np.ones(cfg.max_num_features, dtype=int) * cfg.space_letter
		for xi, i in zip(list(str(val)), np.arange(cfg.max_num_features)):
			x_row[i] = ord(xi)
		return x_row
	
	
	x_data = []
	for x in data['before']:
		x_data.append(q_val(x))
		
	x_data = context_window_transform(x_data, cfg.pad_size)
	
		
	x_data = zip(*x_data)
	
	
	
	
	n = len(x_data)
	
	vals = []
	
	for r, i in zip(x_data, range(n)):
		name = "before_%d"%(i)

		vals.append((name, pd.Series(r)))

	return vals



def extract_after_feature(data, is_normalized=False):
	def q_val(val):
		return val
	vals = data.apply(lambda x: q_val(x['after']), axis=1, raw=True)
	if is_normalized:
		vals = standardization(vals)
	return vals








def extract_all_features(df_list, is_normalized=False):
	data_all = pd.concat(df_list)
	feats_name = [
				'sentence_id',
				'token_id',
# 				'class',
				'before',
# 				'after',
				]

	feats = call_feature_func(data_all, feats_name, is_normalized)
	return feats








def extract_y_info(data):
	return call_feature_func(data, ['class'])





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

def gen_output_vocab():
	df_train = pd.read_csv('../data/en_train.csv')
	df = df_train[df_train['before'] != df_train['after']]
	words = []
	df['after'].apply(lambda x: words.extend(str(x).split(' ')))
	voc = Counter(words).most_common()
	df_out = pd.DataFrame(voc, columns=['word', 'count'])
	df_out.to_csv('../data/out_vocab.csv', index=False)
	
def gen_features(df):
	return None	
	
	
if __name__ == "__main__":
# 	gen_output_vocab()
#  	display_varible_terms()
# 	group_data_address()
# 	num_to_words('105.6',0)
# 	print num2words(105.6)
	display_sentence(78354)
# 	group_data()


