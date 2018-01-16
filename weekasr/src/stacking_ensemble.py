import numpy as np
import pandas as pd
import sys
import tensorflow as tf

from collections import Counter
from keras.utils.np_utils import to_categorical
from tensorflow import keras

sys.path.append('../../')
from config import Config as cfg

cfg.init()


class MajorityVoteEnsembler(object):

	def __init__(self):
		pass

	@staticmethod
	def ensemble(stack_probs, stack_labels, truths=None):
		ensemble_res = []
		join_df = pd.concat(stack_labels, axis=1)
		for i in range(len(join_df)):
			label_count = Counter(join_df.loc[i].values)
			max_cnt = max(label_count.values())
			max_labels = [k for k, v in label_count.items() if v == max_cnt]
			ensemble_res.append(max_labels[0])

		return np.array(ensemble_res)


class AverageProbEnsembler(object):

	def __init__(self):
		pass

	@staticmethod
	def ensemble(stack_probs, stack_labels, truths=None):
		stack_num = len(stack_probs)
		join_df = stack_probs[0]
		for i in range(1, stack_num):
			join_df = join_df + stack_probs[i]
		join_df = join_df / stack_num
		max_labels = join_df.idxmax(1)

		return max_labels.values


class WeightedEnsembler(object):

	def __init__(self, stack_num, cls_num):
		self.learning_rate = 0.001
		self._build_model(stack_num, cls_num)

	def _build_model(self, stack_num, cls_num):
		# Build input and weight.
		self.X = tf.placeholder('float', [None, stack_num * cls_num])
		self.y = tf.placeholder('float', [None, cls_num])
		self.W = tf.Variable(tf.random_normal([stack_num, 1]))

		# Compute loss.
		tile_W = tf.reshape(tf.tile(self.W, [1, cls_num]), (1, -1))
		raw_pred = tf.reshape(tf.multiply(self.X, tile_W), (-1, stack_num, cls_num))
		pred = tf.reduce_mean(raw_pred, axis=1)
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

	def train(self, X_data, y_data, epoch=30, batch_size=256):
		total = len(y_data)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(epoch):
				begin = 0
				while (begin + batch_size < total):
					end = min(begin+batch_size, total)
					batch_X = X_data[begin:end]
					batch_y = y_data[begin:end]
					_, cost = sess.run([self.optimizer, self.cost], feed_dict={self.X:batch_X, self.y:batch_y})
					begin += batch_size
				print 'epoch %d: cost: %.4f' % (epoch , cost)

	def ensemble(stacking_outputs, truths, plabel_list, label_keys):
		pass

def load_data(stack_files, label_names):
	stack_probs = []
	stack_labels = []
	fnames = None
	truths = None
	for f in stack_files:
		df = pd.read_csv(f)
		stack_probs.append(df.loc[:, label_names])
		stack_labels.append(df.loc[:, 'label'])
		if fnames is None:
			fnames = df.loc[:,'fname']
		else:
			assert (fnames.values == df.loc[:,'fname'].values).all()

        if 'truth' in df.columns:
			if truths is None:
				truths = df.loc[:, 'truth'].values
			else:
				assert (truths == df.loc[:,'truth'].values).all()

	if truths is not None:
		for i in range(len(truths)):
			if truths[i] == '<SIL>':
				truths[i] = 'silence'
			elif truths[i] == '<UNK>':
				truths[i] = 'unknown'
			elif truths[i] not in cfg.POSSIBLE_LABELS:
				truths[i] = 'unknown'

	return fnames, truths, stack_probs, stack_labels

def generate_sub_file(fnames, labels, sub_file):
	assert (len(fnames) == len(labels))

	df_sub = pd.DataFrame({'fname':fnames, 'label':labels})
	df_sub = df_sub.replace('<SIL>','silence')
	df_sub = df_sub.replace('<UNK>','unknown')
	df_sub.to_csv(sub_file, index=False)
	print df_sub['label'].value_counts().sort_values()
	print 'Submission samples: %d, file: %s' % (len(fnames), sub_file)

def test_majority(stack_files, sub_file, debug_file=None):
	vocab_labels = cfg.voc_small.dic_w2i.keys()
	fnames, truths, probs_list, labels_list = load_data(
			stack_files, vocab_labels)

	ensembler = MajorityVoteEnsembler()
	adjust_labels = ensembler.ensemble(None, labels_list)

	if debug_file:
		columns = {'fname':fnames}
		for i in range(len(labels_list)):
			columns['stack_%d' % i] = labels_list[i].values
		if truths is not None:
			columns['truth'] = truths
		columns['adjust_label'] = adjust_labels
		df = pd.DataFrame(columns)
		df.to_csv(debug_file, index=False)

	print '---------------- statistics info ---------------'
	# Changed label count.
	for i in range(len(labels_list)):
		num = np.sum(adjust_labels != labels_list[i].values)
		print '  diff count with stack %d: %d' % (i, num)

	# Changed acc.
	if truths is not None:
		samples = len(truths)
		for i in range(len(labels_list)):
			num = np.sum(truths == labels_list[i].values)
			print '  Acc of stack %d: %f (%d/%d)' % (i, num / float(samples), num, samples)
		num = np.sum(adjust_labels == truths)
		print '  Acc of adjust result: %f (%d/%d)' % (num / float(samples), num, samples)
	print '------------------------------------------------'

	if sub_file:
		generate_sub_file(fnames, adjust_labels, sub_file)

def test_avg_prob(stack_files, sub_file, debug_file=None):
	vocab_labels = cfg.voc_small.dic_w2i.keys()
	fnames, truths, probs_list, labels_list = load_data(
			stack_files, vocab_labels)

	ensembler = AverageProbEnsembler()
	adjust_labels = ensembler.ensemble(probs_list, None)

	if debug_file:
		columns = {'fname':fnames}
		if truths is not None:
			columns['truth'] = truths
		for i in range(len(labels_list)):
			columns['stack_%d' % i] = labels_list[i].values
		df = pd.DataFrame(columns)
		df.to_csv(debug_file, index=False)

	print '---------------- statistics info ---------------'
	# Changed label count.
	for i in range(len(labels_list)):
		num = np.sum(adjust_labels != labels_list[i].values)
		print '  Diff count with stack %d: %d' % (i, num)

	# Changed acc.
	if truths is not None:
		samples = len(truths)
		for i in range(len(labels_list)):
			num = np.sum(truths == labels_list[i].values)
			print '  Acc of stack %d: %f (%d/%d)' % (i, num / float(samples), num, samples)
		num = np.sum(adjust_labels == truths)
		print '  Acc of adjust result: %f (%d/%d)' % (num / float(samples), num, samples)
	print '------------------------------------------------'

	if sub_file:
		generate_sub_file(fnames, adjust_labels, sub_file)


stack_test_files = [
	'../sub/liao/stack_training_tf_dsnn_en_d2_1_3_86183.csv',
	'../sub/liao/stack_training_tf_dsnn_en_d2_1_3_86471.csv',
	'../sub/liao/stack_training_tf_dsnn_en_d2_1_3_8745.csv',
	'../sub/liao/stack_training_tf_dsnn_en_d2_3_8555.csv',
	'../sub/liao/stack_training_tf_dsnn_en_d3_1_3_88169.csv',
	'../sub/liao/stack_training_tf_dsnn_en_d3_1_3_88256.csv',
]

stack_train_files = [
	'../ensemble/train_valid_data/stack_training_tf_dsnn_en_d2_1_3_86183_train.csv',
	'../ensemble/train_valid_data/stack_training_tf_dsnn_en_d2_1_3_86471_train.csv',
	'../ensemble/train_valid_data/stack_training_tf_dsnn_en_d2_1_3_8745_train.csv',
	'../ensemble/train_valid_data/stack_training_tf_dsnn_en_d3_1_3_88169_train.csv',
	'../ensemble/train_valid_data/stack_training_tf_dsnn_en_d3_1_3_88256_train.csv',
]

stack_valid_files = [
	'../ensemble/train_valid_data/stack_training_tf_dsnn_en_d2_1_3_86183_valid.csv',
	'../ensemble/train_valid_data/stack_training_tf_dsnn_en_d2_1_3_86471_valid.csv',
	'../ensemble/train_valid_data/stack_training_tf_dsnn_en_d2_1_3_8745_valid.csv',
	'../ensemble/train_valid_data/stack_training_tf_dsnn_en_d3_1_3_88169_valid.csv',
	'../ensemble/train_valid_data/stack_training_tf_dsnn_en_d3_1_3_88256_valid.csv',
]

if __name__ == '__main__':
	# On valid set and output statistics info.
	test_majority(stack_valid_files, None, '../ensemble/major_ensb_valid_debug.csv')
	test_avg_prob(stack_valid_files, None, '../ensemble/avg_prob_ensb_valid_debug.csv')

	# On train set and output statistics info.
	test_majority(stack_train_files, None, '../ensemble/major_ensb_train_debug.csv')
	test_avg_prob(stack_train_files, None, '../ensemble/avg_prob_ensb_train_debug.csv')

	# On test set and create sub file.
	#test_majority(stack_test_files, '../data/major_ensemble_submission.csv', '../ensemble/major_ensb_test_debug.csv')
	#test_avg_prob(stack_test_files, '../data/avg_prob_ensemble_submission.csv', '../ensemble/avg_prob_ensb_test_debug.csv')
