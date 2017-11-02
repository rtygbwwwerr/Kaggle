import data_process
import copy
from Orange.orangeom import star

class Config(object):
	dic_input_char2i = {}
	dic_output_word2i = {}
	dic_output_i2word ={}
	dic_one2one = {}
	output_vocab_size = 0
	dic_constant = None
	input_vocab_size = 0
	vocab_size = 0
	vocab = {}
	vocab_i2word = {}
	
	@staticmethod
	def w2i(w):
		return Config.vocab.get(w, Config.unk_flg_index)
	
	@staticmethod
	def i2w(i):
		return Config.vocab_i2word.get(i, Config.unk_flg)
	
	@staticmethod
	def is2ws(list_i):
		s = [Config.i2w(i) for i in list_i]
		return s
	@staticmethod
	def is2sentence(list_i):
		s = Config.is2ws(list_i)
		" ".join(s)
	@staticmethod
	def init():
		Config.dic_input_char2i = data_process.load_dict_in('../data/input_alpha_table')
		Config.input_vocab_size = len(Config.dic_input_char2i)
		Config.dic_output_word2i = data_process.load_dict('../data/out_vocab.csv')
		Config.output_vocab_size = len(Config.dic_output_word2i)
		Config.vocab = Config.combine(Config.dic_input_char2i, Config.dic_output_word2i)
		Config.vocab_size = len(Config.vocab)
		
		Config.vocab_i2word = {v: k for k, v in Config.vocab.items()}
		Config.dic_output_i2word = {v: k for k, v in Config.dic_output_word2i.items()}
		print 'vocab len total:%d,len w2i:%d, len i2w:%d'%(Config.vocab_size, len(Config.dic_output_word2i), len(Config.dic_output_i2word))
		Config.dic_constant = data_process.load_constant_dict()
		print 'input constant word num:%d'%(len(Config.dic_constant))
		Config.dic_one2one = data_process.load('../data/one2one_dict')
		
	@staticmethod
	def combine(main_dic, dic):
		key0 = set(main_dic.keys())
		key1 = set(dic.keys())
		key_add = key1 - key0
		start = len(key0)
		
		dic_combine = copy.deepcopy(main_dic)
		
		for k in key_add:
			dic_combine[k] = start
			dic[k] = start
			start += 1
		
		
		return dic_combine
		
		
	class1 = set(['PUNCT', 'PLAIN'])
	class2 = set(['VERBATIM', 'LETTERS', 'ELECTRONIC'])
	class3 = set(['FRACTION','TIME','TELEPHONE','DIGIT','MONEY','DECIMAL','ORDINAL','MEASURE','CARDINAL','DATE', 'ADDRESS'])
	batch_size = 100
	max_num_features = 31
# 	max_input_len = 1120,550,280
	max_input_len = 80
	max_classify_input_len = 50
	max_left_input_len = 50
	max_mid_input_len = 50
	max_right_input_len = 50
	embedding_size = 50
	dim_left_embedding = 50
	dim_right_embedding = 50
	dim_mid_embedding = 50
	encoder_max_seq_len = max_input_len
	input_classify_vocab_size = 101
# 	max_output_len = 1850,411
	max_output_len = 111
	max_output_len_decode = max_output_len
	max_token_char_num = 40
	word_vec_len = 255
	max_grad_norm = 5
	learning_rate = 1e-2
	
	save_freq = 10
	eval_freq = 1
	infer_freqv = 1
	n_gpus = 1
	input_hidden_dim = 512
	encoder_hidden_size = 250
	n_encoder_layer = 1
	n_decoder_layer = 1
	decoder_hidden_size = encoder_hidden_size
	space_letter = 0
# 	boundary_word = -3
	pad_size = 1
	boundary_start = 2
	boundary_end = 3
	
	min_char_code = 32
	max_char_code = 126
	
	start_flg = '<GO>'
	end_flg = '<EOS>'
	copy_flg = '<UNK>'
	unk_flg = '<UNK>'
	pad_flg = '<PAD>'
	non_flg = '<PAD>'
	split_input = '<SPLIT-INPUT>'
	split_token = '<SPLIT-TOKEN>'
	non_flg_index = 0
	pad_flg_index = 0
	unk_flg_index = 3
	start_flg_index = 1
	end_flg_index = 2
	copy_flag_index = 3
	start_word = [boundary_start]*(word_vec_len-2)
	end_word = [boundary_end]*(word_vec_len-2)
	
	split_token_index = 4
	split_input_index = 5
 	

	hidden_size = 128 
	drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
	drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
	pool_size = 2
	l2_lambda = 0.0001
	conv_depth = 32 # use 32 kernels in both convolutional layers
	kernel_size = 3
	
	
	data_args_train_no_classify2 = {'df_test':'../data/train_filted_classify.csv',
					   'feat_classify':'../data/en_train_classify.npz',
					   'feat_normalization':'../data/en_train.npz',
					   'model_classify':None,
					   'model_normal':["../checkpoints/l1_l1_c2_weights.39-0.1023-0.9729-0.0981-0.9735.hdf5"],
					   'test_ret_file':'../data/en_train_test_ret.csv',
					   'test_ret_file_err':'../data/en_train_test_ret_err.csv',
					   'ret_file':'../data/en_train_ret.csv',
					   'sub_file':'../data/en_train_submission.csv',
					   'origin_file':"../data/en_train_filted_class.csv",}
	
	data_args_train_no_classify = {'df_test':'../data/train_filted_classify.csv',
					   'feat_classify':'../data/en_train_classify.npz',
					   'feat_normalization':'../data/en_train.npz',
					   'model_classify':None,
					   'model_normal':["../model/g1_g1_c0_weights.37-0.0142-0.9962-0.0127-0.9966.hdf5"],
					   'test_ret_file':'../data/en_train_test_ret.csv',
					   'test_ret_file_err':'../data/en_train_test_ret_err.csv',
					   'ret_file':'../data/en_train_ret.csv',
					   'sub_file':'../data/en_train_submission.csv',
					   'origin_file':"../data/en_train_filted_class.csv",}
	data_args_train = {'df_test':'../data/train_filted_classify.csv',
					   'feat_classify':'../data/en_train_classify.npz',
					   'feat_normalization':'../data/en_train.npz',
					   'model_classify':"../model/class_cnn_c2_weights.98-0.0005-1.0000-0.0004-1.0000.hdf5",
					   'model_normal':["../model/teach_l1_l1_c1_weights.149-0.0022-0.9994-0.0020-0.9995.hdf5", "../model/teach_l1_l1_c2_weights.188-0.0045-0.9988-0.0048-0.9988.hdf5"],
					   'test_ret_file':'../data/en_train_test_ret.csv',
					   'test_ret_file_err':'../data/en_train_test_ret_err.csv',
					   'ret_file':'../data/en_train_ret.csv',
					   'sub_file':'../data/en_train_submission.csv',
					   'origin_file':"../data/en_train_filted_class.csv",}
	
	data_args_test = {'df_test':'../data/test_filted_classify.csv',
					   'feat_classify':'../data/en_test_classify.npz',
					   'feat_normalization':'../data/en_test.npz',
					   'model_classify':"../model/class_cnn_c2_weights.98-0.0005-1.0000-0.0004-1.0000.hdf5",
					   'model_normal':["../model/teach_l1_l1_c1_weights.149-0.0022-0.9994-0.0020-0.9995.hdf5", "../model/teach_l1_l1_c2_weights.188-0.0045-0.9988-0.0048-0.9988.hdf5"],
					   'test_ret_file':'../data/en_train_test_ret.csv',
					   'test_ret_file_err':None,
					   'ret_file':'../data/en_test_ret.csv',
					   'sub_file':'../data/en_submission.csv',
					   'origin_file':"../data/en_test_class.csv",}
	
	dic_month = {"Jan":'january','Feb': 'february','Mar': 'march','Apr': 'april','May': 'may',
			'Jun': 'june','Jul': 'july','Jy': 'july','Aug': 'august','Sep': 'september',
			'Sept': 'september','Oct': 'october','Nov': 'november','Dec' : 'december',
			"1":'january','2': 'february','3': 'march','4': 'april','5': 'may',
			'6': 'june','7': 'july','8': 'august','9': 'september',
			'10': 'october','11': 'november','12' : 'december'}
	dic_week_day = {"Mon":'monday','Monday': 'monday','Tues': 'tuesday','Tuesday': 'tuesday','Wed': 'wednesday',
		'Wednesday': 'wednesday','Thur': 'thursday','Thursday': 'thursday','Fri': 'friday','Friday': 'friday',
		'Saturday': 'saturday','Sat': 'saturday','Sun': 'sunday','Sunday' : 'sunday'}

	special_dic_address = {'I00':'i zero', 
						   'B767':'b seven six seven', 
						   '3':'three',
						   '4':'four',
						    '5':'five', 
						    '6':'six',
						   '7':'seven',
						    '8':'eight', 
						    '9':'nigh',
						   '0':'o'}