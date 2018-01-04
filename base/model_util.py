from base.layers import Seq2SeqBase
from base import data_util
import tensorflow as tf
import numpy as np
import math
import random
from keras.preprocessing import sequence
from base.config_util import Vocab
from base.layers import ClassifierBase
import matplotlib.pyplot as plt
cnames_list = [
'red',
'green',
'blue',
'gold',
'pink',
'purple',
'tomato',
'yellow',
'wheat',
'brown',
'darkgray',
'greenyellow',
'royalblue',
'saddlebrown',
'salmon',
'black',
'aliceblue',
'antiquewhite',
'aqua',
'aquamarine',
'azure',
'beige',
'bisque',
'blanchedalmond',
'blueviolet',
'burlywood',
'cadetblue',
'chartreuse',
'chocolate',
'coral',
'cornflowerblue',
'cornsilk',
'crimson',
'cyan',
'darkblue',
'darkcyan',
'darkgoldenrod',
'darkgreen',
'darkkhaki',
'darkmagenta',
'darkolivegreen',
'darkorange',
'darkorchid',
'darkred',
'darksalmon',
'darkseagreen',
'darkslateblue',
'darkslategray',
'darkturquoise',
'darkviolet',
'deeppink',
'deepskyblue',
'dimgray',
'dodgerblue',
'firebrick',
'floralwhite',
'forestgreen',
'fuchsia',
'gainsboro',
'ghostwhite',
'goldenrod',
'gray',
'honeydew',
'hotpink',
'indianred',
'indigo',
'ivory',
'khaki',
'lavender',
'lavenderblush',
'lawngreen',
'lemonchiffon',
'lightblue',
'lightcoral',
'lightcyan',
'lightgoldenrodyellow',
'lightgreen',
'lightgray',
'lightpink',
'lightsalmon',
'lightseagreen',
'lightskyblue',
'lightslategray',
'lightsteelblue',
'lightyellow',
'lime',
'limegreen',
'linen',
'magenta',
'maroon',
'mediumaquamarine',
'mediumblue',
'mediumorchid',
'mediumpurple',
'mediumseagreen',
'mediumslateblue',
'mediumspringgreen',
'mediumturquoise',
'mediumvioletred',
'midnightblue',
'mintcream',
'mistyrose',
'moccasin',
'navajowhite',
'navy',
'oldlace',
'olive',
'olivedrab',
'orange',
'orangered',
'orchid',
'palegoldenrod',
'palegreen',
'paleturquoise',
'palevioletred',
'papayawhip',
'peachpuff',
'peru',
'plum',
'powderblue',
'rosybrown',
'sandybrown',
'seagreen',
'seashell',
'sienna',
'silver',
'skyblue',
'slateblue',
'slategray',
'snow',
'springgreen',
'steelblue',
'tan',
'teal',
'thistle',
'turquoise',
'violet',
'white',
'whitesmoke',
'yellowgreen']
cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

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


def load_tf_cls_model(sess, model_func, input_info, model_info, model_file_prefix=None):
	
	model = model_func(input_info, model_info)

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
		start_step = (initial_epoch - 1) * (input_info['train_data_num'] / input_info['batch_size'] + 1)
		saver = tf.train.Saver(max_to_keep=100)
		saver.restore(sess, model_file_prefix)
	return model, initial_epoch, start_step	
	
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
	real_ids = interpret_ids(voc, ids)
	strs = join_string.join(voc.i2w(ri) for ri in real_ids)
	return strs

def interpret_ids(voc, ids):
	real_ids = []
	for _id in ids:
		if _id == voc.end_flg_index:
			break
		elif _id != voc.pad_flg_index:
			real_ids.append(_id)
	return real_ids

def eval_result(voc, input_ids, output_ids, infer_output_ids, step, batch_size, print_detil=True):
	right, wrong = 0.0, 0.0
	
	infos = []
# 	print infer_output_ids.shape
	infer_output_ids = infer_output_ids
	for i in range(len(output_ids)):
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

def valid_cls_realtime_data(sess, model, X_list, Y_valid, batch_size, gen_func, 
						func_feature, feature_names, is_augment, is_normalization, batch_num=0, **args):
	mat = None
	accuracy = 0
	num = math.ceil(Y_valid.shape[0] / batch_size)
	for step, data_dict in enumerate(gen_func(X_list, Y_valid, func_feature, feature_names, batch_size, is_augment, is_normalization, False, **args)):
		if (batch_num > 0) and (step > batch_num):
			break
		values = model.run_ops(sess, data_dict, names=["cf_mat", "acc"])
		print "Batch: {}, Accuracy: {}%".format(step, 100 * values['acc'])
		if mat is None:
			mat = values['cf_mat']
		else:
			mat = mat + values['cf_mat']
	
		accuracy = accuracy + values['acc']
	accuracy = accuracy / num
	#plus a little number to prevent overflow
	cls_correct_rate = mat.diagonal() / ((np.sum(mat, axis=1)).astype(np.float32) + 1e-8)
	print mat
	print "class cor-rate:" + str(cls_correct_rate.tolist())
	print "Total: {}, Accuracy: {}%".format(Y_valid.shape[0], accuracy * 100)
	return accuracy, mat, cls_correct_rate

def valid_cls_data(sess, model, X_list, Y_valid, batch_size, gen_func, batch_num=0, **args):
	mat = None
	accuracy = 0
	predict = []
	num = math.ceil(Y_valid.shape[0] / batch_size)
	for step, data_dict in enumerate(gen_func(X_list, Y_valid, batch_size, False, **args)):
		if (batch_num > 0) and (step > batch_num):
			break
		values = model.run_ops(sess, data_dict, names=["cf_mat", "acc", "output"])
		print "Batch: {}, Accuracy: {}%".format(step, 100 * values['acc'])
		if mat is None:
			mat = values['cf_mat']
		else:
			mat = mat + values['cf_mat']
		output = values['output']
# 		print output.shape
		predict.append(output)
	
		accuracy = accuracy + values['acc']
	accuracy = accuracy / num
	#plus a little number to prevent overflow
	cls_correct_rate = mat.diagonal() / ((np.sum(mat, axis=1)).astype(np.float32) + 1e-8)
	predict = np.concatenate(predict)
	print mat
	print "class cor-rate:" + str(cls_correct_rate.tolist())
	print "Total: {}, Accuracy: {}%".format(Y_valid.shape[0], accuracy * 100)
	return predict, accuracy, mat, cls_correct_rate


def valid_seq_data(sess, model, voc, X_valid, Y_valid, PAD_ID_X, PAD_ID_Y, batch_size, gen_func, batch_num=0, **args):
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


def gen_classify_data(X, y, batch_size, shuffle=True):
	number_of_batches = int(math.ceil(y.shape[0] / float(batch_size)))
	counter = 0
	sample_index = np.arange(y.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))

		X_batch = X[batch_index]
		y_batch = y[batch_index]

		counter += 1
		yield X_batch, y_batch
 		if (counter >= number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0

def gen_tf_classify_data(X_list, y, batch_size, shuffle=True, **args):
	number_of_batches = int(math.ceil(y.shape[0] / float(batch_size)))
	counter = 0
	sample_index = np.arange(y.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	data_dict = {}
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))

# 		X_batch = X[batch_index]
		X_batch = map(lambda x: x[batch_index], X_list)
		for i, x in enumerate(X_batch):
			data_dict[ClassifierBase.get_x_name(i)] = x.astype(np.float32)
			
		y_batch = y[batch_index]
		data_dict['Y'] = y_batch
		for k, v in args.iteritems():
			data_dict[k] = v
		counter += 1
		yield data_dict
		if (counter >= number_of_batches):
			break

def gen_tf_realtime_classify_data(X, y, 
								func_gen_features, feature_names, batch_size, is_augment=True, is_normalization=True, 
								shuffle=True, **args):
	number_of_batches = np.ceil(y.shape[0]/batch_size)
	counter = 0

	sample_index = np.arange(y.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	data_dict = {}
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))

# 		X_batch = X[batch_index]
		X_batch = X[batch_index]
		
		X_batch = func_gen_features(X_batch, feature_names, is_augment, is_normalization)
		
		for i, x in enumerate(X_batch):
			data_dict[ClassifierBase.get_x_name(i)] = x.astype(np.float32)
			
		y_batch = y[batch_index]
		data_dict['Y'] = y_batch
		for k, v in args.iteritems():
			data_dict[k] = v
		counter += 1
		yield data_dict
		if (counter >= number_of_batches):
			break

		
def gen_tf_classify_test_data(X_list, batch_size, **args):
	number_of_batches = np.ceil(X_list[0].shape[0]/float(batch_size))
	counter = 0
	sample_index = np.arange(X_list[0].shape[0])

	data_dict = {}
	while True:
		batch_index = get_batch_index(sample_index, batch_size, counter)
		size = len(batch_index)
# 		print size
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))

		X_batch = map(lambda x: x[batch_index], X_list)
		for i, x in enumerate(X_batch):
			data_dict[ClassifierBase.get_x_name(i)] = x
			
		data_dict['Y'] = np.ones((size, 1), np.int32)
		for k, v in args.iteritems():
			data_dict[k] = v
			
		yield data_dict
		counter += 1
		if (counter >= number_of_batches):
			break
def gen_tf_sparse_data(X, y, PAD_ID_X, PAD_ID_Y, batch_size=128, shuffle=True, **args):
	number_of_batches = np.ceil(X.shape[0]/float(batch_size))
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	data_dict = {}
	while True:
		batch_index = get_batch_index(sample_index, batch_size, counter)
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
		if (counter >= number_of_batches):
			break
		
def gen_tf_dense_data(X, y, PAD_ID_X, PAD_ID_Y, batch_size, shuffle, **args):
	number_of_batches = np.ceil(X.shape[0]/float(batch_size))
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	data_dict = {}
	while True:
		batch_index = get_batch_index(sample_index, batch_size, counter)
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))
		X_batch = slice_array(X, batch_index)
		y_batch = slice_array(y, batch_index)
		data_dict['X'] = X_batch
		data_dict['X_lenghts'] = data_util.get_arrays_lengths(X_batch, PAD_ID_X)
		data_dict['Y'] = y_batch
		data_dict['Y_lenghts'] = data_util.get_arrays_lengths(y_batch, PAD_ID_Y)
# 		print data_dict['X'].shape
# 		print data_dict['X_lenghts'].shape
		
		for k, v in args.iteritems():
			data_dict[k] = v
			
		counter += 1
		yield data_dict
		if (counter >= number_of_batches):
			break

def get_batch_index(sample_index, batch_size, counter):
	begin = batch_size*counter
	size = min(batch_size, len(sample_index) - begin)
	end = begin + size
	return sample_index[begin:end]
		
def gen_tf_dense_test_data(X, PAD_ID_X, batch_size, **args):
	number_of_batches = np.ceil(X.shape[0]/float(batch_size))
	counter = 0
	sample_index = np.arange(X.shape[0])

	data_dict = {}
	while True:
		batch_index = get_batch_index(sample_index, batch_size, counter)
		size = len(batch_index)
# 		print size
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))
		X_batch = slice_array(X, batch_index)


		data_dict['X'] = X_batch
		data_dict['X_lenghts'] = data_util.get_arrays_lengths(X_batch, PAD_ID_X)
		data_dict['Y'] = np.ones((size, X.shape[1]), np.int32)
		data_dict['Y_lenghts'] = np.ones((size, ), np.int32)
		for k, v in args.iteritems():
			data_dict[k] = v
			
		
		yield data_dict
		counter += 1
		if (counter >= number_of_batches):
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
					val_ret, mini_acc = valid_seq_data(sess, model, vocab, x_valid, y_valid, PAD_ID_X, PAD_ID_Y, batch_size, data_gen_func, 1, **args)
# 					save_valid_ret(print_step, "mini_valid_ret", val_ret)

					
			epoch_loss = epoch_loss / float(step_nums)
			epoch_acc = epoch_acc / float(step_nums)
			
			val_ret, acc_val = valid_seq_data(sess, model, vocab, x_valid, y_valid, PAD_ID_X, PAD_ID_Y, batch_size, data_gen_func, **args)
	# 		valid_summary_writer.add_summary(summaries_valid, epoch)

# 			print "Val Accuracy: {}%".format(acc_val)


def plot_epoch_in_one(y_list, epoch_range, outdir="../checkpoints/epoch.jpg"):
	'''
	y_list: the values corresponding y axis, and length should be equal to epoch number.
	each item in this list expects form like this:
	([y1,y2,y3,...,yn], color), n = len(epoch_range)
	'''
	plt.subplots()
	plt.xlabel('epoch ids')  
	plt.ylabel('accuracy') 
	plt.grid(True)
# 	plt.figure(figsize=(8, 4))
	for y_info in y_list:
		plt.plot(epoch_range, y_info[0], y_info[1], linewidth = 1)
# 	plt.xlabel("Time(s)")
# 	plt.ylabel("Volt")
# 	plt.title("Line plot")
	plt.savefig(outdir)
	plt.show()
	
	
def plot_epoch(data_path, outdir):
	data = np.loadtxt(data_path)
	data_list = []
	for i in range(data.shape[0]):
		data_list.append((list(data[i]), cnames[cnames_list[i]]))
	plot_epoch_in_one(data_list, range(data.shape[1]), outdir)
	
if __name__ == '__main__':
	plot_epoch()
