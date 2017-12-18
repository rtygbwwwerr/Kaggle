from base.layers import Seq2SeqBase
from base import data_util
import tensorflow as tf
import numpy as np
import math
import random
from keras.preprocessing import sequence
from base.config_util import Vocab





def gen_fake_plus_sequence_data(train_data_num=10000, valid_data_num=1000, scope=10000):
	chars = list('0123456789+')
	dic_w2i = {Vocab.pad_flg:0, Vocab.start_flg:1, Vocab.end_flg:2, Vocab.unk_flg:3}
	start = len(dic_w2i)
	for i, w in enumerate(chars):
		dic_w2i[w] = i+start
	voc = Vocab(dic_w2i)
	
	def int2ids(val):

		ids = []
		for s in str(val):
			ids.append(voc.w2i(s))
		return ids
	
	def gen_data(num, add_start_end=False):
		p = np.random.randint(scope, size=(num, 2))
		arg = None
		sum = None
		if add_start_end:
			arg = map(lambda x:[voc.start_flg_index] + int2ids(x[0]) + [voc.w2i('+')] + int2ids(x[1]) + [voc.end_flg_index], p)
			sum = map(lambda x:[voc.start_flg_index] + int2ids(x[0]+x[1]) + [voc.end_flg_index], p)
		else:
			arg = map(lambda x:int2ids(x[0]) + [voc.w2i('+')] + int2ids(x[1]), p)
			sum = map(lambda x:int2ids(x[0]+x[1]), p)
# 		arg_lens = data_util.get_sequence_lengths(arg)
		arg = sequence.pad_sequences(arg, dtype=np.int32, padding='post', value=0)
		arg = np.reshape(arg, arg.shape+(1,))
# 		sum_lens = data_util.get_sequence_lengths(sum)
		sum = sequence.pad_sequences(sum, dtype=np.int32, padding='post', value=0)
		
		return arg, sum
	
	
	x, y = gen_data(train_data_num)
	x_v, y_v = gen_data(valid_data_num)
	
	return voc, x, y, x_v, y_v
	
	
def load_tf_seq2seq_model(sess, model_func, train_data_num, batch_size, model_file_prefix=None, **arg):

	model = model_func(
			n_encoder_layers = arg['n_encoder_layers'],
			n_decoder_layers = arg['n_decoder_layers'],
			encoder_hidden_size=arg['encoder_hidden_size'], 
			decoder_hidden_size=arg['decoder_hidden_size'], 
			embedding_dim=arg['embedding_dim'], 
			vocab_size=arg['vocab_size'], 
			input_feature_num=arg['input_feature_num'],
			max_decode_iter_size=arg['max_decode_iter_size'],
			PAD = arg['pad_flg_index'],
			START = arg['start_flg_index'],
			EOS = arg['end_flg_index'],
			)

	if model_file_prefix is None:
		start_step = 0
		initial_epoch = 1
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
	else:
		fname = model_file_prefix.split('/')[-1]
		pos_a = fname.find('.')
		pos_b = fname.find('-')
		assert (pos_a != -1 and pos_b != -1)
		initial_epoch = int(fname[pos_a+1:pos_b]) + 1
		start_step = (initial_epoch - 1) * (train_data_num / batch_size + 1)
		saver = tf.train.Saver(max_to_keep=100)
		saver.restore(sess, model_file_prefix)
	return model, initial_epoch, start_step	
	
def interpret(voc, ids, join_string=' '):
	real_ids = []
	for _id in ids:
		if _id == voc.end_flg_index:
			break
		elif _id != voc.pad_flg_index:
			real_ids.append(_id)
			

	return join_string.join(voc.i2w(ri) for ri in real_ids)

def eval_result(voc, input_ids, output_ids, infer_output_ids, step, batch_size, print_detil=True):
	right, wrong = 0.0, 0.0
	
	infos = []
# 	print infer_output_ids.shape
	infer_output_ids = infer_output_ids
	for i in range(batch_size):
		input_sequence = '<DATA>'
		if input_ids is not None:
			input_sequence = interpret(voc, input_ids[i])
		
		output_sequence = interpret(voc, output_ids[i])
		infer_output_sequence = interpret(voc, infer_output_ids[i])
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
	
def valid_data(sess, model, voc, X_valid, Y_valid, PAD_ID_X, PAD_ID_Y, batch_size, gen_func, batch_num=0, **args):
	right = 0
	wrong = 0
	val_ret = []
	for step, data_dict in enumerate(gen_func(X_valid, Y_valid, PAD_ID_X, PAD_ID_Y, batch_size, False, **args)):
		if (batch_num > 0) and (step > batch_num):
			break
		values = model.run_ops(sess, data_dict, names=["search_output"])
# 		print "acc{}:{}".format(step, values['acc'])
		now_right, now_wrong, infos = eval_result(voc, None, data_dict['Y'], values['search_output'], step, batch_size)
		right += now_right
		wrong += now_wrong
		val_ret.extend(infos)
	acc = 100*right/float(right + wrong)
	print "Right: {}, Wrong: {}, Accuracy: {}%".format(right, wrong, acc)
	return val_ret, acc

def slice_array(arr, indeies):
	if arr.dtype == object or len(arr.shape) == 1:
		arr = arr[indeies]
	else:
		arr = arr[indeies, :]
	return arr

def gen_tf_sparse_data(X, y, PAD_ID_X, PAD_ID_Y, batch_size=128, shuffle=True, **args):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	data_dict = {}
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))
		X_batch = slice_array(X, batch_index)
		y_batch = slice_array(y, batch_index)
		data_dict['X'] = X_batch
		data_dict['X_lenghts'] = data_util.get_sequence_lengths(X_batch)
		data_dict['Y_dense'] = y_batch
		data_dict['Y'] = data_util.sparse_tuple_from(y_batch)
		data_dict['Y_lenghts'] = data_util.get_sequence_lengths(y_batch)
		for k, v in args.iteritems():
			data_dict[k] = v
		counter += 1
		yield data_dict
		if (counter == number_of_batches):
			break
		
def gen_tf_dense_data(X, y, PAD_ID_X, PAD_ID_Y, batch_size, shuffle, **args):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	data_dict = {}
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))
		X_batch = slice_array(X, batch_index)
		y_batch = slice_array(y, batch_index)
		data_dict['X'] = X_batch
		data_dict['X_lenghts'] = data_util.get_arrays_lengths(X_batch, PAD_ID_X)
		data_dict['Y'] = y_batch
		data_dict['Y_lenghts'] = data_util.get_arrays_lengths(y_batch, PAD_ID_Y)
		
		
		for k, v in args.iteritems():
			data_dict[k] = v
			
		counter += 1
		yield data_dict
		if (counter == number_of_batches):
			break
		
def gen_tf_dense_test_data(X, PAD_ID_X, batch_size, **args):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])

	data_dict = {}
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))
		X_batch = slice_array(X, batch_index)

		data_dict['X'] = X_batch
		data_dict['X_lenghts'] = data_util.get_arrays_lengths(X_batch, PAD_ID_X)
		
		for k, v in args.iteritems():
			data_dict[k] = v
			
		counter += 1
		yield data_dict
		if (counter == number_of_batches):
			break
	
def test_tf_model(model, vocab, x_train, y_train, x_valid, y_valid, PAD_ID_X, PAD_ID_Y, 
				data_gen_func=gen_tf_sparse_data, batch_size=128, max_epoch=50, **args):
	step_nums = int(math.ceil(x_train.shape[0] / float(batch_size)))
	print "Total epoch:{0}, step num of each epoch:{1}".format(max_epoch, step_nums)
	with tf.Session() as sess:
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		g_step = 1
		print "Training Start!,init step:{}".format(g_step)
		for epoch in range(max_epoch):
			print "Epoch {}".format(epoch)
			epoch_loss = 0.
			epoch_acc = 0.
			for step, data_dict in enumerate(data_gen_func(x_train, y_train, PAD_ID_X, PAD_ID_Y, batch_size, True, **args)):
	
	# 			feed_dict['batch_size'] = data_dict['encoder_input'].shape[0]
	
				values = model.run_ops(sess, data_dict)
				
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
	# 			epoch_acc_seq += values[5]
				
				
				if (print_step) % 300 == 0:
					val_ret, mini_acc = valid_data(sess, model, vocab, x_valid, y_valid, PAD_ID_X, PAD_ID_Y, batch_size, data_gen_func, 1, **args)
# 					save_valid_ret(print_step, "mini_valid_ret", val_ret)

					
			epoch_loss = epoch_loss / float(step_nums)
			epoch_acc = epoch_acc / float(step_nums)
			
			val_ret, acc_val = valid_data(sess, model, vocab, x_valid, y_valid, PAD_ID_X, PAD_ID_Y, batch_size, data_gen_func, **args)
	# 		valid_summary_writer.add_summary(summaries_valid, epoch)

# 			print "Val Accuracy: {}%".format(acc_val)
			
if __name__ == '__main__':
	gen_fake_plus_sequence_data(10, 2, 10)
