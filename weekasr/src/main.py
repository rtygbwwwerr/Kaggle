import pandas as pd
import data_process
import model_maker_keras, model_maker_tf
import numpy as np
from vad import VoiceActivityDetector
from config import Config as cfg
from keras.callbacks import LearningRateScheduler, EarlyStopping,TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.utils.np_utils import to_categorical
import time
import math
import os
import shutil
from tensorflow import keras
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from sklearn.cross_validation import train_test_split
from base import data_util, model_util
from keras.utils import plot_model

cfg.init()

def run_outlier_detector(data_dir, model_path, add_dim=True, model_func=model_maker_keras.make_cnn1):
	vocab = cfg.voc_small
# 	fnames, x = data_process.extract_feature(data_dir, "logspecgram-8000")
	x, fnames = data_process.get_data_from_files("../data/outlier/no_test/", 1.0, ["logspecgram-8000"], ['name'], 0)
	
	if add_dim:
		x = np.reshape(x, x.shape + (1,))
	model = model_func(x[0].shape, vocab.size, trainable=False)
	model.load_weights(model_path)
	print(model.summary())
	
	predict_y = model.predict(x, batch_size = 64, verbose=1)
	p_y = np.argmax(predict_y, axis=1)
	p_labels = map(lambda y:vocab.i2w(y), p_y)
	
	
	
	df_sub = pd.DataFrame({'fnames':fnames,'label':p_labels})
	df_sub = df_sub.replace('<SIL>','silence')
	df_sub = df_sub.replace('<UNK>','unknown')
	
	#replace unknown with silence because no unknown data in 
# 	df_sub['label'] = df_sub['label'].replace('unknown', 'silence')
	file = '../data/outlier/no_outlier.csv'
	print df_sub['label'].value_counts().sort_values()
	df_sub = df_sub[df_sub['label'] == 'unknown']
	print "Outlier samples:%d, file:%s"%(len(df_sub), file)
	df_sub.to_csv(file, index=False)

def run_submission(data_names, add_dim, model_func, vocab, file_type, model_path):
	x, fnames = data_process.get_data_from_files(root_path='../data/test/', data_names=data_names, filter_trim=file_type, pad_id = None)
	if add_dim:
		x = np.reshape(x, x.shape + (1,))
	model = model_func(x[0].shape, vocab.size, trainable=False)
	model.load_weights(model_path)
	print(model.summary())
	submission(model, x, fnames, cfg.voc_small)
	
def run_en_submission(data_names, model_func, vocab, file_type, model_path):
	x, x_wav, fnames = data_process.get_data_from_files(root_path='../data/test/', data_names=data_names, filter_trim=file_type, pad_id = None)
	x = np.reshape(x, x.shape + (1,))
	x_wav = np.reshape(x_wav, x_wav.shape + (1,))
	model = model_func(x[0].shape, x_wav[0].shape, vocab.size, trainable=False)
	model.load_weights(model_path)
	print(model.summary())
	submission(model, [x, x_wav], fnames, '../data/submission_en.csv')
	
def run_submission_cls_tf(vocab, file_head, model_prefix, model_func, data_gen_func=model_util.gen_tf_classify_test_data, input_size=0, batch_size=256):
	
	data_list = data_process.get_data_from_files("../data/test/", cfg.down_rate, cfg.feat_names, cfg.label_test, input_size)
	fnames = data_list[-1]
	x_list = data_list[0:-1]
	
	model_info = {}
	model_info['opt'] = 'adadelta'
 	model_info['model_size_infos'] = cfg.dscnn_model_size_en
	input_info = {}
	input_info['num_cls'] = vocab.size
	input_info['x_dims'] = []
	input_info['batch_size'] = batch_size
	input_info['train_data_num'] = len(fnames)
	input_info['is_training'] = False
	
	print len(data_list)
	for i in range(0, len(data_list) - 1):
		x = data_list[i]
		input_info['x_dims'].append(x[0].shape)
		print "input feature{} shape:{}*{}".format(i, x[0].shape[0], x[0].shape[1])
	
	print "test items num:{}".format(len(fnames))
	

	

	#add new dimension for channels
# 	np.savetxt("../data/X_train.txt", x_train.astype(np.int32), '%d')

# 	x_train = x_train[:, np.newaxis]
# 	x_valid = x_valid[:, np.newaxis]
	p_labels = []
	p_labels_sub = []
	with tf.Session() as sess:
		model, initial_epoch, start_step = model_util.load_tf_cls_model(sess, model_func, input_info, model_info, model_prefix)
		for step, data_dict in enumerate(data_gen_func(x_list, batch_size,
													init_lr_rate = cfg.init_lr_rate, decay_step = cfg.decay_step, 
													decay_factor = cfg.decay_factor, 
													keep_output_rate=1.0, sampling_probability=0.0001)):
			print "processing batch{}".format(step)
			values = model.run_ops(sess, data_dict, names=['output'])
			label = map(lambda x : vocab.i2w(x), values['output'])
			label_sub = map(lambda x :label2sub(x), label)
			p_labels.extend(label)
			p_labels_sub.extend(label_sub)
	df_sub = pd.DataFrame({'fname':fnames, 'label':p_labels_sub})
	df_ret = pd.DataFrame({'fname':fnames, 'label':p_labels_sub, 'plabel':p_labels})
	print df_sub['label'].value_counts().sort_values()
	
	file = '../data/submission_{}.csv'.format(file_head)
	print "Submission samples:%d, file:%s"%(len(fnames), file)
	df_sub.to_csv(file, index=False)
	
	file = '../data/result_{}.csv'.format(file_head)
	print "Test samples:%d, file:%s"%(len(fnames), file)
	df_ret.to_csv(file, index=False)

def label2sub(label):
	out = label
	if label == cfg.sil_flg:
		out = cfg.sil_flg_str
	elif label not in cfg.POSSIBLE_LABELS:
		out = cfg.unk_flg_str
	return out	
		
		
	
def run_submission_seq_tf(file_type, model_prefix, model_func, data_gen_func=model_util.gen_tf_classify_test_data, input_size=0, batch_size=256):
	x, x_wav, fnames = data_process.get_data_from_files(root_path='../data/test/', data_names=["x", "x_wav", "name"], filter_trim=file_type, pad_id = None)
	if input_size > 0:
		x = x[:input_size]
		fnames = fnames[:input_size]
	
	p_labels = []
	p_strs = []
	with tf.Session() as sess:
		model, _, _ = model_util.load_tf_seq2seq_model(sess, model_func, x.shape[0], 
																	batch_size, model_prefix,
																	n_encoder_layers=cfg.n_encoder_layers,
																	n_decoder_layers = cfg.n_decoder_layers,
																	encoder_hidden_size=cfg.encoder_hidden_size,
																	decoder_hidden_size=cfg.decoder_hidden_size,
																	embedding_dim=cfg.embedding_size,
																	vocab_size=cfg.voc_word.size + 1,
																	input_feature_num=x[0].shape[1],
																	is_training=False,
																	max_decode_iter_size=cfg.max_output_len_c + 1,
																	pad_flg_index = cfg.voc_word.pad_flg_index,
																	start_flg_index = cfg.voc_word.start_flg_index,
																	end_flg_index = cfg.voc_word.end_flg_index)
		for step, data_dict in enumerate(data_gen_func(x, None, batch_size,
													init_lr_rate = cfg.init_lr_rate, decay_step = cfg.decay_step, 
													decay_factor = cfg.decay_factor, 
													keep_output_rate=1.0, sampling_probability=0.0001)):
			print "processing batch{}".format(step)
			values = model.run_ops(sess, data_dict, names=['search_output'])
			ids = map(lambda x:model_util.interpret_ids(cfg.voc_char, x), values['search_output'])
			strs = map(lambda x:model_util.interpret(cfg.voc_char, x), ids)
			labels = covert_ids_to_str(cfg.voc_char.i2w, ids)
			p_strs.extend(strs)
			p_labels.extend(labels)
	
	print len(fnames)
	print len(p_labels)
	df_sub = pd.DataFrame({'fname':fnames, 'label':p_labels})
	df_ret = pd.DataFrame({'fname':fnames, 'label':p_labels, 'str':p_strs})
	print df_sub['label'].value_counts().sort_values()
	
	file = '../data/submission_tf.csv'
	print "Submission samples:%d, file:%s"%(len(fnames), file)
	df_sub.to_csv(file, index=False)
	
	file = '../data/result_tf.csv'
	print "Test samples:%d, file:%s"%(len(fnames), file)
	df_ret.to_csv(file, index=False)
	
def extract_tf_model_info(model_prefix, feature_types=cfg.FEATURE_TYPE):
	for type in feature_types:
		type_i_strart = model_prefix.find(type)
		if type_i_strart > 0:
			break
	type_i_end = model_prefix.find(".")
	type = model_prefix[type_i_strart, type_i_end]
	
	epoch_i_start = type_i_end + 1
	epoch_i_end = model_prefix.find("-")
	
	epoch = int(model_prefix[epoch_i_start:epoch_i_end])
	model_type_i_start = model_prefix.find('_') + 1
	model_type_i_end = type_i_strart - 1
	
	mode_type = model_prefix[model_type_i_start:model_type_i_end]
	return type, mode_type, epoch

def covert_ids_to_str(i2w_func, ids):
	"""
	Function behaviors
	1. word in config.POSSIBLE_LABELS, delete <SIL> tags around them and the space inside, like:
	"<SIL> n o <SIL> -> no"
	"<SIL> l e f t <SIL> -> four"
	2. <SIL> sequences, convert to silence:
	"<SIL> <SIL> <SIL> -> silence"
	3. otherwise, convert to unknown:
	"<SIL> f i v e <SIL> -> unknown"
	"<SIL> f o u r <SIL> -> unknown"
	
	
	Args:
	i2w_func, i2w function, input id and output respective word or char, ref:config_util.Vocab.i2w()
	ids, numpy array of word ids, which includes ids <SIL> and common chars, like aforementioned in Function behaviors.
	
	
	Return:
	converted string
	"""
	
	def convert(ids):
		chars = map(lambda id:i2w_func(id), ids)
		word = ''.join(chars)
		if word == cfg.sil_flg:
			word = 'silence'
		elif word not in cfg.POSSIBLE_LABELS:
			word = 'unknown'
		return word

	decode_str = map(lambda x:convert(x[1:-1]), ids)
	
	
	return decode_str

def format_labels(l):
	new_label = l
	if l not in cfg.voc_small.dic_w2i:
		new_label = cfg.unk_word
	return new_label
	
def submission(model, x, fnames, vocab, file = '../data/submission.csv'):
	
	predict_y = model.predict(x, batch_size = 256, verbose=0)
	p_y = np.argmax(predict_y, axis=1)
	p_labels = map(lambda y:vocab.i2w(y), p_y)
	
	
	
	df_sub = pd.DataFrame({'fname':fnames, 'label':p_labels})
	df_sub = df_sub.replace('<SIL>','silence')
	df_sub = df_sub.replace('<UNK>','unknown')
	#replace unknown with silence because no unknown data in 
# 	df_sub['label'] = df_sub['label'].replace('unknown', 'silence')
	
	print df_sub['label'].value_counts().sort_values()
	print "Submission samples:%d, file:%s"%(len(fnames), file)
	df_sub.to_csv(file, index=False)

def classify_generator_ensemble(X, x_wav, y, batch_size=128, shuffle=True):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))

		X_batch = X[batch_index,:]
		X_wav_batch = x_wav[batch_index,:]
		y_batch = y[batch_index,:]
		
		# reshape X to be [samples, time steps, features]
# 		for i, j in enumerate(batch_index):
# 			tmpx = X[j]
# 			for t in range(X.shape[1]):
#  				X_batch[i,t,tmpx[t]] = 1.0
		
		counter += 1
		yield [X_batch, X_wav_batch], y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0
			

			
def capsule_classify_generator(X, y, batch_size=128, shuffle=True):
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
		yield ([X_batch, y_batch], [y_batch, X_batch])
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0


		


def train_keras_model(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, batch_size=128, initial_epoch=0, nb_epoch = 0):
	

	
	board = TensorBoard(log_dir='../logs/', histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head)
	checkpointer = ModelCheckpoint(monitor="val_acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
 	
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
# 	samples_per_epoch = batch_size * 2
	model.fit_generator(generator=model_util.gen_classify_data(X_train, Y_train, batch_size, True), 
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
	
def train_keras_ensemble(model, ret_file_head, data_dict, batch_size=128, initial_epoch=0, nb_epoch = 0):
	
	x_train = data_dict['x_train']
	x_wav_train = data_dict['x_wav_train']
	y_train = data_dict['y_train']
	x_valid = data_dict['x_valid']
	x_wav_valid = data_dict['x_wav_valid']
	y_valid = data_dict['y_valid']
	
	board = TensorBoard(log_dir='../logs/', histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head)
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
 	
	samples_per_epoch = int(math.ceil(x_train.shape[0] / float(batch_size)))
# 	samples_per_epoch = batch_size * 2
	model.fit_generator(generator=classify_generator_ensemble(x_train, x_wav_train, y_train, batch_size, True), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
	                    validation_data = ([x_valid, x_wav_valid], y_valid),
# 			    	    validation_data=sparse_generator(X_valid, Y_valid, batch_size, False), 
# 			    	    nb_val_samples=int(math.ceil(X_valid.shape[0] / float(batch_size))),
			    	    callbacks=[board, checkpointer]
			    		)
	print 'Training time', time.time() - start_time
	# evaluate network
	score = model.evaluate([x_valid, x_wav_valid], y_valid, batch_size)

# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)
	
def train_keras_capsule_model(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, batch_size=128, nb_epoch = 3):
	

	
	board = keras.callbacks.TensorBoard(log_dir='../logs/', batch_size=batch_size, histogram_freq=0)
	check_file = "../checkpoints/%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head)
	checkpointer = keras.callbacks.ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
 	
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
# 	samples_per_epoch = batch_size * 2
	model.fit_generator(generator=capsule_classify_generator(X_train, Y_train, batch_size, True), 
	                    steps_per_epoch = samples_per_epoch, 
	                    epochs = nb_epoch, 
	                    verbose=1,
	                    validation_data = ([X_valid, Y_valid], [Y_valid, X_valid]),
# 			    	    validation_data=sparse_generator(X_valid, Y_valid, batch_size, False), 
# 			    	    nb_val_samples=int(math.ceil(X_valid.shape[0] / float(batch_size))),
			    	    callbacks=[board, checkpointer, model_maker_keras.make_lr_decay()]
			    		)
	print 'Training time', time.time() - start_time
	# evaluate network
	score = model.evaluate(X_valid, Y_valid, batch_size)

# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)


def interpret(i2str, ids, eos_id=-1, join_string=' '):
	real_ids = []
	for _id in ids:
		if _id != eos_id:
			real_ids.append(_id)
		else:
			break

	return join_string.join(i2str(ri) for ri in real_ids)

def eval_result(i2str, pad_flg, input_ids, output_ids, infer_output_ids, step, batch_size, print_detil=True):
	right, wrong = 0.0, 0.0
	
	infos = []
	for i in range(batch_size):
		input_sequence = '<DATA>'
		if input_ids is not None:
			input_sequence = ' '.join(interpret(i2str, input_ids[i]).replace(pad_flg, '').split())
		
		output_sequence = ' '.join(interpret(i2str, output_ids[i]).replace(pad_flg, '').split())
		infer_output_sequence = interpret(i2str, infer_output_ids[i])
		info = 'EVAL:{}==>{} -> {} (Real: {})'.format(step * batch_size + i, input_sequence, infer_output_sequence, output_sequence)
		try:
			if output_sequence.strip() == infer_output_sequence.strip():
				info = "{}:{}".format("[Right]", info)
				right += 1.0
			else:
				info = "{}:{}".format("[False]", info)
				wrong += 1.0
		except ValueError:  # output_sequence == ''
			wrong += 1.0
		if print_detil:
			print info
		infos.append(info)
# 	print "Right: {}, Wrong: {}, Accuracy: {}%".format(right, wrong, 100*right/float(right + wrong))
	return right, wrong, infos
	

def sigmoid_increase(x, change_step):
	x = (x / change_step) * change_step
# 	val = (1 / (1 + math.exp(13 - 1.0 / 10000*x)) + 1 / (1 + math.exp(8 - 1.0 / 5000*x)) * (1/7.0)) / (1.1428571428571428)
# 	val = (1 / (1 + math.exp(11 - 1.0 / 10000*x)) + 1 / (1 + math.exp(8 - 1.0 / 5000*x)) * (1/6.0)) / (1.05)
	val = (1 / (1 + math.exp(10 - 1.0 / 7500*x)) + 1 / (1 + math.exp(8 - 1.0 / 5000*x)) * (1/5.0)) / (1.2)
	return val
	
		
def save_valid_ret(index, name, val_ret):
	path = "../data/{}_{}.txt".format(name, index)
	fp = open(path, "w")
	for line in val_ret:
		fp.write('%s\n' % line)
	fp.close()	

def train_tf_model(
			data_gen_func, vocab, sess, model, log_dir, ret_file_head, 
			X_train, Y_train, X_valid, Y_valid, PAD_ID_X, PAD_ID_Y, 
			initial_epoch, start_step, batch_size=128, nb_epoch = 100):

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
	g_step = start_step
	print "Training Start!,init step:{}".format(g_step)

	for epoch in range(initial_epoch, max_epoch):
		print "Epoch {}".format(epoch)
		epoch_loss = 0.
		epoch_acc = 0.
		epoch_acc_seq = 0.
		for step, data_dict in enumerate(data_gen_func(X_train, Y_train, PAD_ID_X, PAD_ID_Y, batch_size, True, 
													init_lr_rate = cfg.init_lr_rate, decay_step = cfg.decay_step, decay_factor = cfg.decay_factor, 
													keep_output_rate=cfg.keep_output_rate, sampling_probability=0.0001)):
# 			print g_step
# 			feed_dict['batch_size'] = data_dict['encoder_input'].shape[0]
			values = model.run_ops(sess, data_dict, names=["train_op", 'output', 'search_output','summary', 'global_step', "loss", "acc", "seq_acc", "sp"])
			
			g_step = values['global_step']
			print_step = step + 1
	
			info_head= "Epoch:{:3d}/{:3d}, g_Step:{:8d}, Step:{:6d}/{:6d} ... ".format(epoch, max_epoch, g_step, step + 1, step_nums)
			info_vals = []
			for name, value in values.iteritems():
# 				print type(value)
				if data_util.isDigitType(value) and name != 'global_step':
					info_vals.append("{}: {}".format(name, value))
			
			info_vals = ", ".join(info_vals)
			
			print info_head + info_vals
			epoch_loss += values['loss']
			epoch_acc += values['acc']
			epoch_acc_seq += values["seq_acc"]

			train_summary_writer.add_summary(values['summary'], g_step)
			
			if print_step % 100 == 0:
				now_right, now_wrong, _ = model_util.eval_result(vocab, None, data_dict['Y'], values['search_output'], 0, batch_size)
				acc = 100*now_right/float(now_right + now_wrong)
				print "Current Right: {}, Wrong: {}, Accuracy: {}%".format(now_right, now_wrong, acc)

				
		epoch_loss = epoch_loss / float(step_nums)
		epoch_acc = epoch_acc / float(step_nums)
		epoch_acc_seq = epoch_acc_seq / float(step_nums)
		val_ret, acc_val = model_util.valid_seq_data(sess, model, vocab, X_valid, Y_valid, PAD_ID_X, PAD_ID_Y, batch_size, data_gen_func,
												 init_lr_rate = cfg.init_lr_rate, decay_step = cfg.decay_step, decay_factor = cfg.decay_factor,
												 keep_output_rate=cfg.keep_output_rate, sampling_probability=0.0)
# 		valid_summary_writer.add_summary(summaries_valid, epoch)
		path = "../checkpoints/tf/{prefix}.{epoch_id:02d}-{loss:.5f}-{acc:.5f}-{val_acc:.5f}.ckpt".format(prefix=ret_file_head,
																					    epoch_id=epoch,
																					    loss=epoch_loss,
																					    acc=epoch_acc_seq,
																					    val_acc=(acc_val/100.0),
																					    )
		saved_path = saver.save(sess, path, global_step=model.get_op('global_step'))
		print "saved check file:" + saved_path
# 		print "Val Accuracy: {:.2}%".format(acc_val)

def train_tf_classifier(
			data_gen_func, vocab, sess, model, log_dir, ret_file_head, 
			train_x_list, Y_train, fid_train, valid_x_list, Y_valid, fid_valid,
			initial_epoch, start_step, batch_size=128, nb_epoch = 100):

	max_epoch = nb_epoch
	step_nums = int(math.ceil(Y_train.shape[0] / float(batch_size)))
	print "Total epoch:{}, batch size:{}, step num of each epoch:{}".format(max_epoch, batch_size, step_nums)

	saver = tf.train.Saver(max_to_keep=1000)
# 	summary_writer = tf.train.summary.SummaryWriter('../log/tf/', sess.graph)
	train_summary_writer = tf.summary.FileWriter(log_dir, sess.graph_def)
	g_step = start_step
	print "Training Start!,init step:{}".format(g_step)
	max_acc = 0.0
	val_accs = []
	epoch_accs = []
	epoch_losses = []
	epoch_cls_accs = []
	for epoch in range(initial_epoch, max_epoch):
		print "Epoch {}".format(epoch)
		epoch_loss = 0.
		epoch_acc = 0.
		for step, data_dict in enumerate(data_gen_func(train_x_list, Y_train, batch_size, True, 
													init_lr_rate = cfg.init_lr_rate, decay_step = cfg.decay_step, decay_factor = cfg.decay_factor, 
													dropout_prob=cfg.keep_output_rate)):
# 			print g_step
# 			feed_dict['batch_size'] = data_dict['encoder_input'].shape[0]
			values = model.run_ops(sess, data_dict, names=["train_op",'summary', 'global_step', "loss", "acc", "lr"])
			
			g_step = values['global_step']
	
			info_head= "Epoch:{:3d}/{:3d}, g_Step:{:8d}, Step:{:6d}/{:6d} ... ".format(epoch, max_epoch - 1, g_step, step + 1, step_nums)
			info_vals = []
			for name, value in values.iteritems():
# 				print type(value)
				if data_util.isDigitType(value) and name != 'global_step':
					info_vals.append("{}: {}".format(name, value))
			
			info_vals = ", ".join(info_vals)
			
			print info_head + info_vals
			epoch_loss += values['loss']
			epoch_acc += values['acc']

			train_summary_writer.add_summary(values['summary'], g_step)

				
		epoch_loss = epoch_loss / float(step_nums)
		epoch_acc = epoch_acc / float(step_nums)
		epoch_accs.append(epoch_acc)
		epoch_losses.append(epoch_loss)
		print "avg_loss:{}, avg_acc:{}".format(epoch_loss, epoch_acc)
		predict, acc_val, mat, cls_rate = model_util.valid_cls_data(sess, model, valid_x_list, Y_valid, batch_size, data_gen_func,
												 init_lr_rate = cfg.init_lr_rate, decay_step = cfg.decay_step, decay_factor = cfg.decay_factor,
												 keep_output_rate=cfg.keep_output_rate, sampling_probability=0.0)
# 		valid_summary_writer.add_summary(summaries_valid, epoch)
		save_val_ret(predict, fid_valid, vocab)
		np.savetxt("../data/con_mat_{}.txt".format(epoch), mat.astype(np.int32), '%d', delimiter='	')
		epoch_cls_accs.append(cls_rate)
		display_cls_rate_info(cls_rate, vocab)
		
		val_accs.append(acc_val)
		
		save_epoch_info(ret_file_head, epoch_accs, val_accs, epoch_cls_accs)
		if acc_val > max_acc:
			path = "../checkpoints/tf/{prefix}.{epoch_id:02d}-{loss:.5f}-{acc:.5f}-{val_acc:.5f}.ckpt".format(prefix=ret_file_head,
																						    epoch_id=epoch,
																						    loss=epoch_loss,
																						    acc=epoch_acc,
																						    val_acc=acc_val,
																						    )
			saved_path = saver.save(sess, path, global_step=model.get_op('global_step'))
			print "val_acc imporved from {} to {}, saved check file:{}".format(max_acc, acc_val, saved_path)
			max_acc = acc_val
		else:
			print "val_acc does not improve!skip"
			
def save_val_ret(predict, fid_valid, vocab, outdir="../checkpoints/epoch_ret.csv"):
	predict = map(lambda x: vocab.i2w(x), predict)
	print len(fid_valid)
	print len(predict)
	df = pd.DataFrame({'fname':fid_valid, 'label':predict})
	print "save evaluate samples:%d, file:%s"%(len(fid_valid), outdir)
	df.to_csv(outdir, index=False)
	
		
def train_tf_realtime_classifier(
			data_gen_func, vocab, sess, model, log_dir, ret_file_head, 
			X_train, Y_train, X_vaild, Y_valid,
			initial_epoch, start_step, batch_size=128, nb_epoch = 100):

	max_epoch = nb_epoch
	step_nums = int(math.ceil(Y_train.shape[0] / float(batch_size)))
	print "Total epoch:{}, batch size{}, step num of each epoch:{}".format(max_epoch, batch_size, step_nums)

	saver = tf.train.Saver(max_to_keep=1000)
# 	summary_writer = tf.train.summary.SummaryWriter('../log/tf/', sess.graph)
	train_summary_writer = tf.summary.FileWriter(log_dir, sess.graph_def)
	g_step = start_step
	print "Training Start!,init step:{}".format(g_step)
	max_acc = 0.0
	val_accs = []
	epoch_accs = []
	epoch_losses = []
	for epoch in range(initial_epoch, max_epoch):
		print "Epoch {}".format(epoch)
		epoch_loss = 0.
		epoch_acc = 0.
		for step, data_dict in enumerate(data_gen_func(X_train, Y_train, data_process.gen_features_realtime, cfg.feat_names, batch_size, True, True, True,
													init_lr_rate = cfg.init_lr_rate, decay_step = cfg.decay_step, decay_factor = cfg.decay_factor, 
													dropout_prob=cfg.keep_output_rate)):
# 			print g_step
# 			feed_dict['batch_size'] = data_dict['encoder_input'].shape[0]
			values = model.run_ops(sess, data_dict, names=["train_op",'summary', 'global_step', "loss", "acc", "lr"])
			
			g_step = values['global_step']
	
			info_head= "Epoch:{:3d}/{:3d}, g_Step:{:8d}, Step:{:6d}/{:6d} ... ".format(epoch, max_epoch - 1, g_step, step + 1, step_nums)
			info_vals = []
			for name, value in values.iteritems():
# 				print type(value)
				if data_util.isDigitType(value) and name != 'global_step':
					info_vals.append("{}: {}".format(name, value))
			
			info_vals = ", ".join(info_vals)
			
			print info_head + info_vals
			epoch_loss += values['loss']
			epoch_acc += values['acc']

			train_summary_writer.add_summary(values['summary'], g_step)

				
		epoch_loss = epoch_loss / float(step_nums)
		epoch_acc = epoch_acc / float(step_nums)
		epoch_accs.append(epoch_acc)
		epoch_losses.append(epoch_loss)
		print "avg_loss:{}, avg_acc:{}".format(epoch_loss, epoch_acc)
		acc_val, mat, cls_rate = model_util.valid_cls_realtime_data(sess, model, X_vaild, Y_valid, batch_size, data_gen_func, data_process.gen_features_realtime,
													cfg.feat_names, False,	True,
												 init_lr_rate = cfg.init_lr_rate, decay_step = cfg.decay_step, decay_factor = cfg.decay_factor,
												 keep_output_rate=cfg.keep_output_rate, sampling_probability=0.0)
# 		valid_summary_writer.add_summary(summaries_valid, epoch)
		np.savetxt("../data/con_mat_{}.txt".format(epoch), mat.astype(np.int32), '%d', delimiter='	')
		display_cls_rate_info(cls_rate, vocab)
		
		val_accs.append(acc_val)
		
		save_epoch_info(ret_file_head, epoch_accs, val_accs)
		if acc_val > max_acc:
			path = "../checkpoints/tf/{prefix}.{epoch_id:02d}-{loss:.5f}-{acc:.5f}-{val_acc:.5f}.ckpt".format(prefix=ret_file_head,
																						    epoch_id=epoch,
																						    loss=epoch_loss,
																						    acc=epoch_acc,
																						    val_acc=acc_val,
																						    )
			saved_path = saver.save(sess, path, global_step=model.get_op('global_step'))
			print "val_acc imporved from {} to {}, saved check file:{}".format(max_acc, acc_val, saved_path)
			max_acc = acc_val
		else:
			print "val_acc does not improve!skip"
			
			
def save_epoch_info(file_head, *values):
	arr_list = map(lambda x : np.array(x).reshape(1, len(x), -1), values)
	for arr in arr_list:
		print arr.shape
		
	
	data = np.concatenate(arr_list, 2)
	data = np.squeeze(data)
	data = data.T
	np.savetxt("../checkpoints/{}_epoch.txt".format(file_head), data.astype(np.float32), '%f', delimiter='	')
		
def display_cls_rate_info(cls_rate, vocab):
	for i, rate in enumerate(cls_rate):
		word = vocab.i2w(i)
		print "{}-{}".format(word, rate)
			
def restore_tf_model(model_prefix, sess, input_shape, model_func, is_train=True):
	# Load graph file.
	saver = tf.train.import_meta_graph(model_prefix + '.meta')
	saver.restore(sess, model_prefix)

	# Create model.
	model = model_func(
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
	# Restore parameters in Model object with restored value.
	model.restore_from_session(sess)

	return model

def experiment_keras_outlier(train_func=train_keras_model, model_func=model_maker_keras.make_cnn1, 
					batch_size=64, nb_epoch=50, input_num=0, add_dim=True, 
					file_head="keras_outlier_no", pre_train_model=None):
	
	x_neg, y_neg = data_process.get_data_from_files("../data/outlier/neg/", 1.0, ["logspecgram-8000"], ['simple'], input_num)
	x_pos, y_pos = data_process.get_data_from_files("../data/outlier/pos/", 1.0, ["logspecgram-8000"], ['simple'], input_num)
	y_pos[:] = cfg.voc_small.w2i(cfg.unk_flg)
	y_neg[:] = cfg.voc_small.w2i('no')
	x = np.vstack([x_neg, x_pos])
	y = np.concatenate([y_neg, y_pos])
	
	x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=batch_size*2, random_state=0)
	
	print "Training data info:"
	count_value_info(y_train)
	
	print "Valid data info:"
	count_value_info(y_valid)
	
	y_train = to_categorical(y_train.tolist(), cfg.voc_small.size)
	y_valid = to_categorical(y_valid.tolist(), cfg.voc_small.size)
	np.savetxt("../data/Y_train.txt", y_train.astype(np.int32), '%d')
	#add new dimension for channels
	if add_dim :
		x_train=np.reshape(x_train,x_train.shape + (1,))
		x_valid=np.reshape(x_valid,x_valid.shape + (1,))
# 	x_train = x_train[:, np.newaxis]
# 	x_valid = x_valid[:, np.newaxis]
	print "train feature shape:{}*{}".format(x_train[0].shape[0], x_train[0].shape[1])
	print "train items num:{0}, valid items num:{1}".format(x_train.shape[0], x_valid.shape[0])
	model = model_func(x_train[0].shape, len(y_train[0]))
	print(model.summary())
	initial_epoch = 0
	if pre_train_model:
		model.load_weights(pre_train_model)
		index_start = pre_train_model.find("_weights.") + len('_weights.')
		index_end = pre_train_model.find("-")
		initial_epoch = int(pre_train_model[index_start:index_end])
	
	train_func(model, file_head, x_train, y_train, x_valid, y_valid, batch_size, initial_epoch, initial_epoch + nb_epoch)
def experiment_keras(train_func=train_keras_model, model_func=model_maker_keras.make_cnn1, 
					batch_size=128, nb_epoch=50, input_num=0, add_dim=True, data_names=['x', 'y'],
					file_head="keras_cnn7", file_type="mfcc", pre_train_model=None):
	
	x_train, y_train = data_process.get_data_from_files("../data/train/", data_names, file_type, input_num)
	x_valid, y_valid = data_process.get_data_from_files("../data/valid/", data_names, file_type, input_num)
	y_train = to_categorical(y_train.tolist(), cfg.voc_small.size)
	y_valid = to_categorical(y_valid.tolist(), cfg.voc_small.size)
	np.savetxt("../data/Y_train.txt", y_train.astype(np.int32), '%d')
	#add new dimension for channels
	if add_dim :
		x_train=np.reshape(x_train,x_train.shape + (1,))
		x_valid=np.reshape(x_valid,x_valid.shape + (1,))
# 	x_train = x_train[:, np.newaxis]
# 	x_valid = x_valid[:, np.newaxis]
	print "train feature shape:{}*{}".format(x_train[0].shape[0], x_train[0].shape[1])
	print "train items num:{0}, valid items num:{1}".format(x_train.shape[0], x_valid.shape[0])
	model = model_func(x_train[0].shape, len(y_train[0]))
	print(model.summary())
	initial_epoch = 0
	if pre_train_model:
		model.load_weights(pre_train_model)
		index_start = pre_train_model.find("_weights.") + len('_weights.')
		index_end = pre_train_model.find("-")
		initial_epoch = int(pre_train_model[index_start:index_end])
	
	train_func(model, file_head, x_train, y_train, x_valid, y_valid, batch_size, initial_epoch, initial_epoch + nb_epoch)

def cut_around(y):
	if y.shape[1] > 1:
		return y[:, 1:-1]
	else:
		return y

def experiment_keras_ensemble(train_func=train_keras_model, model_func=model_maker_keras.make_cnn_en1, vocab=cfg.voc_small,
					batch_size=128, nb_epoch=50, input_num=0, add_dim=True, data_names=['x', 'x_wav', 'y'],
					file_head="keras_cnn7", file_type="mfcc", pre_train_model=None):
	
	x_train, x_wav_train, y_train = data_process.get_data_from_files("../data/train/", data_names, file_type, input_num)
	x_valid, x_wav_valid, y_valid = data_process.get_data_from_files("../data/valid/", data_names, file_type, input_num)
	y_train = cut_around(y_train)
	y_valid = cut_around(y_valid)
	y_train = to_categorical(y_train.tolist(), vocab.size)
	y_valid = to_categorical(y_valid.tolist(), vocab.size)
	np.savetxt("../data/Y_train.txt", y_train.astype(np.int32), '%d')
	#add new dimension for channels
	if add_dim :
		x_train=np.reshape(x_train,x_train.shape + (1,))
		x_valid=np.reshape(x_valid,x_valid.shape + (1,))
		x_wav_train=np.reshape(x_wav_train,x_wav_train.shape + (1,))
		x_wav_valid=np.reshape(x_wav_valid,x_wav_valid.shape + (1,))
		
	print "train feature shape:{}*{}".format(x_train[0].shape[0], x_train[0].shape[1])
	print "train items num:{0}, valid items num:{1}".format(x_train.shape[0], x_valid.shape[0])
	
	data_dict = {}
	data_dict['x_train'] = x_train
	data_dict['x_wav_train'] = x_wav_train
	data_dict['y_train'] = y_train
	data_dict['x_valid'] = x_valid
	data_dict['x_wav_valid'] = x_wav_valid
	data_dict['y_valid'] = y_valid

	model = model_func(x_train[0].shape, x_wav_train[0].shape, len(y_train[0]))
	print(model.summary())
	initial_epoch = 0
	if pre_train_model:
		model.load_weights(pre_train_model)
		index_start = pre_train_model.find("_weights.") + len('_weights.')
		index_end = pre_train_model.find("-")
		initial_epoch = int(pre_train_model[index_start:index_end])
		
	plot_model(model, show_shapes=True, to_file='../pic/{}.png'.format(file_head))
	train_func(model, file_head, data_dict, batch_size, initial_epoch, initial_epoch + nb_epoch)


def experiment_tf(data_gen_func, train_func=train_tf_model, model_func=model_maker_tf.make_CapsNet, label_name='y', 
					batch_size=256, nb_epoch=50, input_num=0, 
					file_head="tf_cap3", file_type="mfcc", pre_train_model_prefix=None, is_debug=False):
	
	x_train, y_train, x_valid, y_valid = data_process.data_prepare(input_num, file_head, file_type, label_name)
	

	
	#add new dimension for channels
# 	np.savetxt("../data/X_train.txt", X_train.astype(np.int32), '%d')
# 	np.savetxt("../data/Y_train.txt", Y_train.astype(np.int32), '%d')
# 	x_train = x_train[:, np.newaxis]
# 	x_valid = x_valid[:, np.newaxis]
	
	
	with tf.Session() as sess:
		# Create model or load pre-trained model.
		if is_debug:
			sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		model = None
		start_step = 0
		if pre_train_model_prefix is None:
			model = model_func(cfg.CLS_NUM, x_train.shape[1], x_train.shape[2], batch_size, True)
			initial_epoch = 1
			sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		else:
			initial_epoch = 0
			start_step = (initial_epoch - 1) * (x_train.shape[0] / batch_size + 1)
			model = restore_tf_model(pre_train_model_prefix, sess, x_train.shape, model_func)
			
		log_dir = '../logs/tf/'
	
		ret_file_head = "{}_{}".format(file_head, file_type)
		train_func(data_gen_func, sess, model, log_dir, ret_file_head, 
			x_train, y_train, x_valid, y_valid, 
			initial_epoch, start_step, batch_size, nb_epoch)
	
def count_value_info(y):
	N = len(y)
	print "Total items:{}".format(N)
	values, cnts = np.unique(y, return_counts=True)
	for value, cnt in zip(values, cnts):
		print "val :{}={} :{}".format(value, cnt, cnt / float(N))


def init_cls_weight(vocab, ws = 3.0):
	vec = np.ones(vocab.size)
	labels_small = cfg.voc_small.wordset()
	labels_all = vocab.wordset()
	dlabels = labels_all - labels_small
	rate = ws / len(labels_small)
	for label in labels_small:
		vec[vocab.w2i(label)] = 1.0 + rate
	print "Class Weight vector:"
	print vec
# 	for label in dlabels:
# 		if label != cfg.pad_flg and label != cfg.end_flg and label != cfg.start_flg:
# 			vec[vocab.w2i(label)] = 1.0 - rate
			
	return vec.astype(np.float32)
		
def experiment_tf_classifier(vocab, data_gen_func, train_func=train_tf_classifier, model_func=model_maker_tf.make_tf_dscnn,
					batch_size=256, nb_epoch=50, input_num=0, 
					file_head="tf_dscnn", pre_train_model_prefix=None, is_debug=False):
	
	x_num = len(cfg.feat_names)
	data_train = data_process.get_data_from_files("../data/train/", cfg.down_rate, cfg.feat_names, cfg.label_names, input_num)
# 	print len(data_train)
	data_valid = data_process.get_data_from_files("../data/valid/", cfg.down_rate, cfg.feat_names, cfg.label_names, input_num)
# 	data_list = train_test_split(*data_train, test_size=batch_size*10, random_state=0)
	train_x_list = data_train[0:-2]
	print len(train_x_list)
	valid_x_list = data_valid[0:-2]
	y_train = data_train[-2]
	y_valid = data_valid[-2]
	fid_train = data_train[-1]
	fid_valid = data_valid[-1]
 	
 	print y_valid.shape
# 	y_train = data_list[-4]
# 	y_valid = data_list[-3]
# 	fid_train = data_list[-2]
# 	fid_valid = data_list[-1]
	print "Training data info:"
	count_value_info(y_train)
	
	print "Valid data info:"
	count_value_info(y_valid)
	np.savetxt("../data/Y_train_org.txt", y_train.astype(np.int32), '%d')
	np.savetxt("../data/Y_valid_org.txt", y_valid.astype(np.int32), '%d')
# 	y_train = np.reshape(y_train, (y_train.shape[0], ))
# 	y_valid = np.reshape(y_valid, (y_valid.shape[0], ))
	y_train = to_categorical(y_train.tolist(), vocab.size)
	y_valid = to_categorical(y_valid.tolist(), vocab.size)
	
	np.savetxt("../data/Y_train.txt", y_train.astype(np.int32), '%d')
	np.savetxt("../data/Y_valid.txt", y_valid.astype(np.int32), '%d')
	
	model_info = {}
	model_info['opt'] = 'adadelta'
 	model_info['model_size_infos'] = cfg.dscnn_model_size_en
#  	model_info['cls_weight'] = init_cls_weight(vocab)
#	model_info['model_size_info'] = [6, 176, 10, 4, 2, 1, 176, 3, 3, 2, 2, 176, 3, 3, 1, 1, 176, 3, 3, 1, 1, 176, 3, 3, 1, 1, 176, 3, 3, 1, 1]
	input_info = {}
	input_info['num_cls'] = vocab.size
	input_info['x_dims'] = []
	input_info['batch_size'] = batch_size
	input_info['train_data_num'] = y_train.shape[0]
	input_info['is_training'] = True
	
	train_x_list = []
	valid_x_list = []

# 	for i in xrange(0, len(data_list) - 4, 2):
# 		x_train = data_list[i]
# 		x_vaild = data_list[i + 1]
# 		train_x_list.append(x_train)
# 		valid_x_list.append(x_vaild)
# 		input_info['x_dims'].append(x_train[0].shape)
# 		print "input feature{} shape:{}*{}".format(i/2, x_train[0].shape[0], x_train[0].shape[1])

	for i in xrange(0, len(data_train) - 2, 1):
		x_train = data_train[i]
		x_vaild = data_valid[i]
		train_x_list.append(x_train)
		valid_x_list.append(x_vaild)
		input_info['x_dims'].append(x_train[0].shape)
		print "input feature{} shape:{}*{}".format(i, x_train[0].shape[0], x_train[0].shape[1])	
		
	print "train items num:{0}, valid items num:{1}".format(y_train.shape[0], y_valid.shape[0])
	

	

	#add new dimension for channels
# 	np.savetxt("../data/X_train.txt", x_train.astype(np.int32), '%d')

# 	x_train = x_train[:, np.newaxis]
# 	x_valid = x_valid[:, np.newaxis]
	with tf.Session() as sess:
		if is_debug:
			sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		model, initial_epoch, start_step = model_util.load_tf_cls_model(sess, model_func, input_info, model_info, pre_train_model_prefix)
	
			
		log_dir = '../logs/tf/'
		ret_file_head = "{}_{}".format(file_head,  cfg.feat_names[0])
		train_func(data_gen_func, vocab, sess, model, log_dir, ret_file_head, 
			train_x_list, y_train, fid_train, valid_x_list, y_valid, fid_valid,
			initial_epoch, start_step, batch_size, initial_epoch + nb_epoch)

def experiment_tf_realtime_classifier(data_gen_func, train_func=train_tf_realtime_classifier, model_func=model_maker_tf.make_tf_dscnn,
					batch_size=256, nb_epoch=50, input_num=0, 
					file_head="tf_dscnn_rl", pre_train_model_prefix=None, is_debug=False):
	
	vocab = cfg.voc_small

	data_train = data_process.get_data_from_files("../data/train/", cfg.down_rate, ['rawwav'], cfg.label_names, input_num)
# 	print len(data_train)
	data_valid = data_process.get_data_from_files("../data/valid/", cfg.down_rate, ['rawwav'], cfg.label_names, input_num)
#  	data_list = train_test_split(*data_train, test_size=batch_size*20, random_state=0)
 	x_train = data_train[0]
 	x_valid = data_valid[0]
 	y_train = data_train[-1]
 	y_valid = data_valid[-1]
 	
#  	y_train = data_list[-2]
#  	y_valid = data_list[-1]
	print "Training data info:"
	count_value_info(y_train)
	
	print "Valid data info:"
	count_value_info(y_valid)
	np.savetxt("../data/Y_train_org.txt", y_train.astype(np.int32), '%d')
	np.savetxt("../data/Y_valid_org.txt", y_valid.astype(np.int32), '%d')
# 	y_train = np.reshape(y_train, (y_train.shape[0], ))
# 	y_valid = np.reshape(y_valid, (y_valid.shape[0], ))
	y_train = to_categorical(y_train.tolist(), vocab.size)
	y_valid = to_categorical(y_valid.tolist(), vocab.size)
	
	np.savetxt("../data/Y_train.txt", y_train.astype(np.int32), '%d')
	np.savetxt("../data/Y_valid.txt", y_valid.astype(np.int32), '%d')
	
	model_info = {}
 	model_info['model_size_infos'] = cfg.dscnn_model_size_en
#  	model_info['cls_weight'] = init_cls_weight(vocab)
#	model_info['model_size_info'] = [6, 176, 10, 4, 2, 1, 176, 3, 3, 2, 2, 176, 3, 3, 1, 1, 176, 3, 3, 1, 1, 176, 3, 3, 1, 1, 176, 3, 3, 1, 1]
	input_info = {}
	input_info['num_cls'] = vocab.size
	input_info['x_dims'] = []
	input_info['batch_size'] = batch_size
	input_info['train_data_num'] = y_train.shape[0]
	input_info['is_training'] = True
	input_info['x_dims'].append((99, 40))
# 	train_x_list = []
# 	valid_x_list = []
# 
# 	for i in xrange(0, len(data_list) - 2, 2):
# 		x_train = data_list[i]
# 		x_vaild = data_list[i + 1]
# 		train_x_list.append(x_train)
# 		valid_x_list.append(x_vaild)
# 		input_info['x_dims'].append(x_train[0].shape)
# 		print "input feature{} shape:{}*{}".format(i/2, x_train[0].shape[0], x_train[0].shape[1])

# 	for i in xrange(0, len(data_train) - 1, 1):
# 		x_train = data_train[i]
# 		input_info['x_dims'].append((99, 40))
# 		print "input feature{} shape:{}*{}".format(i, x_train[0].shape[0], x_train[0].shape[1])	
		
	print "train items num:{0}, valid items num:{1}".format(y_train.shape[0], y_valid.shape[0])
	

	

	#add new dimension for channels
# 	np.savetxt("../data/X_train.txt", x_train.astype(np.int32), '%d')

# 	x_train = x_train[:, np.newaxis]
# 	x_valid = x_valid[:, np.newaxis]
	with tf.Session() as sess:
		if is_debug:
			sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		model, initial_epoch, start_step = model_util.load_tf_cls_model(sess, model_func, input_info, model_info, pre_train_model_prefix)
	
			
		log_dir = '../logs/tf/'
		ret_file_head = "{}_{}".format(file_head,  cfg.feat_names[0])
		train_func(data_gen_func, vocab, sess, model, log_dir, ret_file_head, 
			x_train, y_train, x_valid, y_valid,
			initial_epoch, start_step, batch_size, initial_epoch + nb_epoch)

def experiment_tf_seq2seq(data_gen_func, train_func=train_tf_model, model_func=model_maker_tf.make_tf_AttentionSeq2Seq, label_name='y_w', 
					batch_size=256, nb_epoch=50, input_num=0, 
					file_head="tf_seq2seq", file_type="mfcc", pre_train_model_prefix=None, is_debug=False):
	
	x_train, y_train = data_process.get_data_from_files("../data/train/", ['x', label_name], file_type, input_num)
	x_valid, y_valid = data_process.get_data_from_files("../data/valid/", ['x', label_name], file_type, input_num)
	file_head = "{}_{}".format(file_head, file_type)
	
	print "train feature shape:{}*{}".format(x_train[0].shape[0], x_train[0].shape[1])
	print "train items num:{0}, valid items num:{1}".format(x_train.shape[0], x_valid.shape[0])

	
	x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=batch_size*20, random_state=0)
	
	
	print "Real train items num:{0}, valid items num:{1}".format(x_train.shape[0], x_valid.shape[0])
	#add new dimension for channels
# 	np.savetxt("../data/X_train.txt", x_train.astype(np.int32), '%d')
	np.savetxt("../data/Y_train.txt", y_train.astype(np.int32), '%d')
	np.savetxt("../data/Y_valid.txt", y_valid.astype(np.int32), '%d')
# 	x_train = x_train[:, np.newaxis]
# 	x_valid = x_valid[:, np.newaxis]
	with tf.Session() as sess:
		if is_debug:
			sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		model, initial_epoch, start_step = model_util.load_tf_seq2seq_model(sess, model_func, x_train.shape[0], 
																	batch_size, pre_train_model_prefix,
																	n_encoder_layers=cfg.n_encoder_layers,
																	n_decoder_layers = cfg.n_decoder_layers,
																	encoder_hidden_size=cfg.encoder_hidden_size,
																	decoder_hidden_size=cfg.decoder_hidden_size,
																	embedding_dim=cfg.embedding_size,
																	vocab_size=cfg.voc_word.size + 1,
																	input_feature_num=x_train[0].shape[1],
																	max_decode_iter_size=cfg.max_output_len_c + 1,
																	pad_flg_index = cfg.voc_word.pad_flg_index,
																	start_flg_index = cfg.voc_word.start_flg_index,
																	end_flg_index = cfg.voc_word.end_flg_index)
	
			
		log_dir = '../logs/tf/'
		ret_file_head = "{}_{}".format(file_head, file_type)
		train_func(data_gen_func, cfg.get_vocab(label_name), sess, model, log_dir, ret_file_head, 
			x_train, y_train, x_valid, y_valid, None, cfg.get_vocab(label_name).pad_flg_index,
			initial_epoch, start_step, batch_size, initial_epoch + nb_epoch)

def detect_file(filename):
# 	filename = '/Users/user/wav-sample.wav'
	v = VoiceActivityDetector(filename)
	v.plot_detected_speech_regions()

def copy(label='silence', extid=4):
	output = "../data/train/ext{}/".format(extid)
	if not os.path.exists(output):
		os.makedirs(output)
	
	df = pd.read_csv("../data/submission.csv")
	df_out = df[df['label']==label]
	df_out['fname'].apply(lambda x: shutil.copyfile("../data/test/audio/" + x, output + x))
	print "completed copy {} {} files.".format(len(df_out), label)
	
	
def test_ctc_model():
	batch_size = 64
	vocab, x_train, y_train, x_valid, y_valid = model_util.gen_fake_plus_sequence_data(train_data_num=10000, valid_data_num=64, scope=10)
	model = model_maker_tf.make_tf_CTCSeq2seq(
			n_encoder_layers = cfg.n_encoder_layers,
			n_decoder_layers = cfg.n_decoder_layers,
# 					dropout = cfg.ed_dropout,
			encoder_hidden_size=cfg.encoder_hidden_size, 
			decoder_hidden_size=cfg.decoder_hidden_size, 
			batch_size=batch_size, 
			embedding_dim=cfg.embedding_size, 
			vocab_size=vocab.size + 1, 
			input_feature_num=x_train.shape[2],
			max_decode_iter_size=cfg.max_output_len_w,
			PAD = cfg.voc_word.pad_flg_index,
			START = cfg.voc_word.start_flg_index,
			EOS = cfg.voc_word.end_flg_index,
			)
	
	model_util.test_tf_model(model, vocab, x_train, y_train, x_valid, y_valid, batch_size=batch_size, max_epoch=3300, 
							init_lr_rate=1e-4, decay_step=1500, decay_factor=0.85, keep_output_rate=0.9)

def test_attention_model():
	batch_size = 64
	vocab, x_train, y_train, x_valid, y_valid = model_util.gen_fake_plus_sequence_data(train_data_num=10000, valid_data_num=batch_size, scope=40)
	
	np.savetxt("../data/X_train.txt", x_train.astype(np.int32), '%d')
	np.savetxt("../data/Y_train.txt", y_train.astype(np.int32), '%d')
	np.savetxt("../data/X_valid.txt", x_valid.astype(np.int32), '%d')
	np.savetxt("../data/Y_valid.txt", y_valid.astype(np.int32), '%d')
	
	data_analysis(x_train, y_train, x_valid, y_valid)
	
	model = model_maker_tf.make_tf_AttentionSeq2Seq(
			n_encoder_layers = cfg.n_encoder_layers,
			n_decoder_layers = cfg.n_decoder_layers,
# 					dropout = cfg.ed_dropout,
			encoder_hidden_size=cfg.encoder_hidden_size, 
			decoder_hidden_size=cfg.decoder_hidden_size, 
			embedding_dim=cfg.embedding_size, 
			vocab_size=vocab.size + 1, 
			input_feature_num=x_train.shape[2],
			max_decode_iter_size=3,
			PAD = vocab.pad_flg_index,
			START = vocab.start_flg_index,
			EOS = vocab.end_flg_index,
			)
	
	model_util.test_tf_model(model, vocab, x_train, y_train, x_valid, y_valid, PAD_ID_X=None, PAD_ID_Y=None, 
							batch_size=batch_size, data_gen_func=model_util.gen_tf_dense_data, max_epoch=10000, 
							init_lr_rate=1.0, decay_step=5000, decay_factor=0.85, keep_output_rate=1.0, sampling_probability=0.0001)
	
def data_analysis(x_train, y_train, x_valid, y_valid):
	

	cnt_x = 0
	cnt_all = 0
	for i, x in enumerate(x_valid):
		for j, xt in enumerate(x_train):
			if np.array_equal(x, xt):
				cnt_x += 1
				if np.array_equal(y_valid[i], y_train[j]):
					cnt_all += 1
				break
		
	
	eqx_rate = cnt_x / float(x_valid.shape[0])	
	eq_rate = cnt_all / float(x_valid.shape[0])
	print "annlysis result: eqx_rate:{}, eq_rate:{}".format(eqx_rate, eq_rate)

def result_analysis(sub_file="../data/ds_submission.csv"):
	df_ret = pd.read_csv(sub_file)
	print df_ret['label'].value_counts().sort_values()
	
# 	flags = np.sum(flags, axis=1, dtype=np.int32)
# 	
# 	flags_index = np.argwhere(flags==x_train.shape[1])
# 	
# 	
# 	eq_rate = len(flags_index) / float(x_valid.shape[0])
# 	
# 	print eq_rate
	
def find_best_model_file_pre(root_path, file_head):
	trim = ".data-00000-of-00001"
	files = data_process.gen_input_paths(root_path, file_beg_name=file_head, file_ext_name=trim)
	max_acc = 0.0
	best_file = None
	for file in files:
		index_end = file.find('.ckpt')
		index_begin = file.rfind('-', 0, index_end) + 1
		val_acc = float(file[index_begin:index_end])
		if val_acc > max_acc:
			max_acc = val_acc
			best_file = file[0:file.find(trim)]
	max_acc_str = str(max_acc)
	max_acc_str = max_acc_str[max_acc_str.find('0.') + 2:]
	if best_file is not None:
		print "find best model:" + best_file
	return best_file, max_acc_str
if __name__ == "__main__":
# 	test_attention_model()
# 	test_ctc_model()
# 	data_process.test()
# 	detect_file("../data/outlier/up/9f869f70_nohash_1-up.wav")
# 	copy('no', extid=4)
# 	copy('up', extid=5)
# 	copy('stop', extid=6)
# 	copy('unknown', extid=7)
	ftype = "logspecgram_8000"
# 	ftype = 'mfbank_16000'
# 	ftype = "logbank"
# 	ftype = "mfcc40_16000"
	
# 	vocab = cfg.voc_small
# 	result_analysis("../data/crnn_9500_submission.csv")

# 	experiment_keras(train_func=train_keras_model, model_func=model_maker_keras.make_cnn1,
# 					 batch_size=256, nb_epoch=100, input_num=0, file_head="keras_cnn", file_type=ftype)

# 	experiment_keras(train_func=train_keras_model, model_func=model_maker_keras.make_rnn1, add_dim=False, 
# 					 batch_size=256, nb_epoch=100, input_num=0, file_head="keras_rnn_gru", file_type=ftype)
	
# 	experiment_keras(train_func=train_keras_model, model_func=model_maker_keras.make_dscnn1,
# 					 batch_size=256, nb_epoch=100, input_num=0, file_head="keras_dscnn", file_type=ftype)
# 	experiment_keras(train_func=train_keras_model, model_func=model_maker_keras.make_cnn2, add_dim=True, data_names=['x_wav', 'y'],
# 					 batch_size=256, nb_epoch=100, input_num=0, file_head="keras_cnn_wav", file_type=ftype)
	
# 	experiment_keras_ensemble(train_func=train_keras_ensemble, model_func=model_maker_keras.make_cnn_en2, vocab=vocab,
# 						batch_size=256, nb_epoch=100, input_num=0, add_dim=True, data_names=['x', 'x_wav', 'y'],
# 						file_head="keras_cnn_en", file_type=ftype, pre_train_model=None)
# 	experiment_tf(train_func=train_tf_model, model_func=model_maker_tf.make_CapsNet, label_name='y',
# 					batch_size=64, nb_epoch=50, input_num=0, 
# 					file_head="tf_cap3", file_type=ftype, pre_train_model_prefix=None)
	
	
# 	experiment_tf_seq2seq(data_gen_func=model_util.gen_tf_sparse_data, train_func=train_tf_model, model_func=model_maker_tf.make_tf_CTCSeq2seq, label_name='y_w',
# 					batch_size=256, nb_epoch=50, input_num=0, 
# 					file_head="tf_ctc_seq2seq", file_type=ftype, pre_train_model_prefix=None)
	
# 	experiment_tf_seq2seq(data_gen_func=model_util.gen_tf_dense_data, train_func=train_tf_model, model_func=model_maker_tf.make_tf_AttentionSeq2Seq, label_name='y_c',
# 					batch_size=256, nb_epoch=150, input_num=0, 
# 					file_head="tf_att_seq2seq", 
# 					file_type=ftype, pre_train_model_prefix="../checkpoints/tf/tf_att_seq2seq_mfcc_16000.27-0.05072-0.96616-0.97383.ckpt-7506", is_debug=False)
	
# 	experiment_tf_seq2seq(data_gen_func=model_util.gen_tf_dense_data, train_func=train_tf_model, model_func=model_maker_tf.make_tf_AttentionSeq2Seq, label_name='y_c',
# 					batch_size=256, nb_epoch=300, input_num=0, 
# 					file_head="tf_att_seq2seq", 
# 					file_type=ftype, pre_train_model_prefix=None, is_debug=False)
	file_head = "tf_dscnn"
	max_acc_str = None
	model_pre = None
	vocab = cfg.voc_small
# 	experiment_keras_outlier()
# 	run_outlier_detector('../data/train/audio/no/', '../checkpoints/keras_outlier_no_weights.26-0.0015-0.9998-0.0110-1.0000.hdf5')
	model_pre, max_acc_str = find_best_model_file_pre("../checkpoints/tf/", file_head)
	experiment_tf_classifier(vocab, model_util.gen_tf_classify_data, train_func=train_tf_classifier, model_func=model_maker_tf.make_tf_dscnn, 
					batch_size=256, nb_epoch=50, input_num=0, 
					file_head=file_head, pre_train_model_prefix=model_pre, is_debug=False)
# 	model_util.plot_epoch("../checkpoints/{}_mfcc40s_epoch.txt".format(file_head), "../checkpoints/epoch_no_outlier_3.jpg")
# 	model_pre, max_acc_str = find_best_model_file_pre("../checkpoints/tf/", file_head)
# 	file_head = file_head + "_" + max_acc_str
# # # 	
# 	run_submission_cls_tf(vocab, file_head, model_pre, model_maker_tf.make_tf_dscnn, input_size=0, batch_size=256)

# 	model_util.plot_epoch("../checkpoints/tf_dscnn_en_l_mfcc40s_epoch.txt", "../checkpoints/epoch_en_l.jpg")
# 	model_util.plot_epoch("../checkpoints/tf_dscnn_en_w_mfcc40s_epoch.txt", "../checkpoints/epoch_en_w.jpg")
# 	run_submission_tf(file_type=ftype, input_size = 100,
# 					model_prefix="../model/tf/tf_att_seq2seq_logspecgram_8000.166-0.00080-0.99590-0.94902.ckpt-42828", 
# 					model_func=model_maker_tf.make_tf_AttentionSeq2Seq)
# 	run_submission(['x', 'names'], False, model_maker_keras.make_rnn1, vocab, ftype, '../checkpoints/keras_rnn_gru_weights.24-0.1139-0.9580-0.1305-0.9477.hdf5')
# 	run_en_submission(['x', 'x_wav', 'names'], model_func=model_maker_keras.make_cnn_en2, vocab=vocab, file_type=ftype, model_path='../checkpoints/keras_cnn_en_weights.18-0.0077-0.9972-0.0956-0.9762.hdf5')
# 	detect_file('../data/test/audio/clip_0667aa08a.wav')




