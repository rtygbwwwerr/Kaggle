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
cfg.init ()


def run_submission(file_type="mfcc", model_path='../checkpoints/keras_cnn4_logbank_weights.14-0.0724-0.9788-0.2771-0.9415.hdf5'):
	x = data_process.get_test_data_from_files(filter_trim=file_type)
	x = np.reshape(x, x.shape + (1,))
	model = model_maker_keras.make_cnn1(x[0].shape, cfg.voc_small.size, trainable=False)
	model.load_weights(model_path)
	submission(model, x)
	
def run_submission_tf(file_type, model_prefix, model_func, data_gen_func, batch_size=256):
	x = data_process.get_test_data_from_files(filter_trim=file_type)
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
																	max_decode_iter_size=cfg.max_output_len_c + 1,
																	pad_flg_index = cfg.voc_word.pad_flg_index,
																	start_flg_index = cfg.voc_word.start_flg_index,
																	end_flg_index = cfg.voc_word.end_flg_index)
	for step, data_dict in enumerate(data_gen_func(x, cfg.voc_word.pad_flg_index, batch_size,
												init_lr_rate = cfg.init_lr_rate, decay_step = cfg.decay_step, 
												decay_factor = cfg.decay_factor, 
												keep_output_rate=cfg.keep_output_rate, sampling_probability=0.0001)):
# 		feed_dict = model.make_null_feed_dict(data_dict)
		values = model.run_ops(sess, data_dict, names=['search_output'])


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
	
	
	
	decode_str = None
	return decode_str
		
def submission(model, x, file = '../data/submission.csv'):
	
	paths = data_process.gen_input_paths("../data/test/audio/", ".wav")
	fnames = map(lambda f:os.path.basename(f), paths)
	
	predict_y = model.predict(x, batch_size = 256, verbose=0)
	p_y = np.argmax(predict_y, axis=1)
	p_labels = map(lambda y:cfg.i2n(y), p_y)
	
	
	
	df_sub = pd.DataFrame({'fname':fnames, 'label':p_labels})
	df_sub = df_sub.replace('<SIL>','silence')
	df_sub = df_sub.replace('<UNK>','unknown')
	#replace unknown with silence because no unknown data in 
# 	df_sub['label'] = df_sub['label'].replace('unknown', 'silence')
	
	print df_sub['label'].value_counts().sort_values()
	print "Submission samples:%d, file:%s"%(len(fnames), file)
	df_sub.to_csv(file, index=False)

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


		


def train_keras_model(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, batch_size=128, nb_epoch = 3):
	

	
	board = TensorBoard(log_dir='../logs/', histogram_freq=0, write_graph=True,
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
# 			if (print_step) % 300 == 0:
# 				val_ret, mini_acc = model_util.valid_data(sess, model, vocab, X_valid, Y_valid, PAD_ID_X, PAD_ID_Y, batch_size, data_gen_func, 1,
# 														 init_lr_rate = cfg.init_lr_rate, decay_step = cfg.decay_step, decay_factor = cfg.decay_factor,
# 														 keep_output_rate=cfg.keep_output_rate, sampling_probability=0.0001)
# 				save_valid_ret(print_step, "mini_valid_ret", val_ret)
# 				print "Mini-test Accuracy: {:.2}%".format(mini_acc)
# 				path = "../checkpoints/mini/{prefix}.{epoch_id}_{id:02d}-{val_acc:.5f}.ckpt".format(
# 																								prefix="mini",
# 																							    epoch_id=epoch,
# 																							    id=print_step,
# 																							    val_acc=mini_acc,
# 																							    )
# 				saved_path = saver.save(sess, path, global_step=print_step)
# 				print "saved check file:" + saved_path
				
		epoch_loss = epoch_loss / float(step_nums)
		epoch_acc = epoch_acc / float(step_nums)
		epoch_acc_seq = epoch_acc_seq / float(step_nums)
		val_ret, acc_val = model_util.valid_data(sess, model, vocab, X_valid, Y_valid, PAD_ID_X, PAD_ID_Y, batch_size, data_gen_func,
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

def experiment_keras(train_func=train_keras_model, model_func=model_maker_keras.make_cnn1, 
					batch_size=128, nb_epoch=50, input_num=0, 
					file_head="keras_cnn7", file_type="mfcc", pre_train_model=None):
	
	x_train, y_train, x_valid, y_valid = data_process.data_prepare(input_num, file_head, file_type)
	y_train = to_categorical(y_train.tolist(), cfg.voc_small.size)
	y_valid = to_categorical(y_valid.tolist(), cfg.voc_small.size)
	np.savetxt("../data/Y_train.txt", y_train.astype(np.int32), '%d')
	#add new dimension for channels
	x_train=np.reshape(x_train,x_train.shape + (1,))
	x_valid=np.reshape(x_valid,x_valid.shape + (1,))
# 	x_train = x_train[:, np.newaxis]
# 	x_valid = x_valid[:, np.newaxis]
	
	model = model_func(x_train[0].shape, len(y_train[0]))
	print(model.summary())
	initial_epoch = 0
	if pre_train_model:
		model.load_weights(pre_train_model)
		index_start = pre_train_model.find("_weights.") + len('_weights.')
		index_end = pre_train_model.find("-")
		initial_epoch = int(pre_train_model[index_start:index_end])
	
	train_func(model, file_head, x_train, y_train, x_valid, y_valid, batch_size, initial_epoch, initial_epoch + nb_epoch)

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



def experiment_tf_seq2seq(data_gen_func, train_func=train_tf_model, model_func=model_maker_tf.make_tf_AttentionSeq2Seq, label_name='y_w', 
					batch_size=256, nb_epoch=50, input_num=0, 
					file_head="tf_seq2seq", file_type="mfcc", pre_train_model_prefix=None, is_debug=False):
	
	x_train, y_train, x_valid, y_valid = data_process.data_prepare(input_num, file_head, file_type, label_name)
	
	x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=batch_size*20, random_state=0)

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
	
# 	flags = np.sum(flags, axis=1, dtype=np.int32)
# 	
# 	flags_index = np.argwhere(flags==x_train.shape[1])
# 	
# 	
# 	eq_rate = len(flags_index) / float(x_valid.shape[0])
# 	
# 	print eq_rate
	
	

if __name__ == "__main__":
# 	test_attention_model()
# 	test_ctc_model()
# 	data_process.test()
	
# 	copy('no', extid=4)
# 	copy('up', extid=5)
# 	copy('stop', extid=6)
# 	copy('unknown', extid=7)
	ftype = "logspecgram_8000"
# 	ftype = 'mfbank_16000'
# 	ftype = "logbank"
	ftype = "mfcc_16000"
# 	experiment_keras(train_func=train_keras_model, model_func=model_maker_keras.make_cnn1,
# 					 batch_size=256, nb_epoch=100, input_num=0, file_head="keras_cnn", file_type=ftype)
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
	
	experiment_tf_seq2seq(data_gen_func=model_util.gen_tf_dense_data, train_func=train_tf_model, model_func=model_maker_tf.make_tf_AttentionSeq2Seq, label_name='y_c',
					batch_size=256, nb_epoch=300, input_num=0, 
					file_head="tf_att_seq2seq", 
					file_type=ftype, pre_train_model_prefix=None, is_debug=False)
	
# 	run_submission(file_type=ftype, model_path='../checkpoints/keras_cnn_weights.42-0.0110-0.9960-0.0048-0.9987.hdf5')
# 	detect_file('../data/test/audio/clip_0667aa08a.wav')




