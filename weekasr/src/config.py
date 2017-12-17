from base.config_util import Vocab
class Config(object):
	
	
	POSSIBLE_LABELS = 'yes no up down left right on off stop go'.split()
	
	voc_char = None
	voc_word = None
	voc_small = None
	
	CLS_NUM = 0
	
	
	sil_flg = '<SIL>'
	unk_flg = '<UNK>'
	pad_flg = Vocab.pad_flg
	start_flg = Vocab.start_flg
	end_flg = Vocab.end_flg

	init_dic = {pad_flg:0, start_flg:1, end_flg:2, unk_flg:3, sil_flg:4}

	min_char_code = 32
	max_char_code = 126
	sampling_rate = 16000
	
	
	init_lr_rate=1.0
	decay_step=2000
	decay_factor=0.85
	
	ed_keep_rate = 0.9
	de_keep_rate = 1.0
	keep_output_rate = 1.0
	
	sp_decay_step = 1000

	n_encoder_layers = 2
	n_decoder_layers = 1
	encoder_hidden_size = 128
	decoder_hidden_size = encoder_hidden_size
	embedding_size = 10

	max_output_len_c = 12
	max_output_len_w = 3
	
	dict_vocabs = {}
	
	
	@staticmethod
	def get_vocab(label_name):
		return Config.dict_vocabs[label_name]
	
	@staticmethod
	def init():
		Config.voc_char = Vocab(Config.get_char_dict())
		Config.voc_word = Vocab(Config.get_word_dict())
		Config.voc_small = Vocab(Config.get_word_small_dict())
		Config.CLS_NUM = Config.voc_small.size
		
		Config.dict_vocabs['y'] = Config.voc_small
		Config.dict_vocabs['y_c'] = Config.voc_char
		Config.dict_vocabs['y_w'] = Config.voc_word
	@staticmethod
	def i2c(i):
		return Config.voc_char.i2w(i)
	
	@staticmethod
	def c2i(w):
		return Config.voc_char.w2i(w)
		
	@staticmethod
	def i2w(i):
		return Config.voc_word.i2w(i)
	
	@staticmethod
	def w2i(w):
		return Config.voc_word.w2i(w)
		
	@staticmethod
	def i2n(i):
		return Config.voc_small.i2w(i)
	
	@staticmethod
	def n2i(w):
		return Config.voc_small.w2i(w)
	
	@staticmethod
	def get_char_dict():
		file = open("../data/dict", "r")
		dic = Config.init_dic.copy()
		
		start = len(dic)
		char_set = set()
		lines = file.readlines()
		for line in lines:
			line = line.strip()
			if line.startswith("<") and line.endswith('>'):
				char_set.add(line)
			else:
				char_set = char_set | set(line)
	
		chars = list(char_set)
		chars.sort()
		for k, v in enumerate(chars):
			dic[v] = k + start
			
		
		list_table = []
		list_min = []
		list_max = []
		for c in char_set:
			if ord(c) >= Config.min_char_code and ord(c) <= Config.max_char_code:
				list_table.append(ord(c))
			elif ord(c) < Config.min_char_code:
				list_min.append(ord(c))
			else:
				list_max.append(ord(c))
		print "char num in [0,31]:%d"%(len(list_table))
		print "char num in [32,126]:%d"%len(list_min)
		print "char num in [127,):%d"%len(list_max)
		print "char num total:%d"%len(dic)
		print "load vocab_char"
		print dic
		return dic
	
	@staticmethod
	def get_word_small_dict():
		dic_w2i = {name: i for i, name in enumerate(Config.POSSIBLE_LABELS)}
		dic_w2i[Config.unk_flg] = len(dic_w2i)
		dic_w2i[Config.sil_flg] = len(dic_w2i)
		print "load vocab_small"
		print dic_w2i
		return dic_w2i
	
	@staticmethod
	def get_word_dict():
		file = open("../data/dict", "r")
		dic = Config.init_dic.copy()
		start = len(dic)
		word_set = set()
		lines = file.readlines()
		for line in lines:
			line = line.strip()
			word_set.add(line)
			
		words = list(word_set)
		words.sort()
		for k, v in enumerate(words):
			dic[v] = k + start
		print "word num total:%d"%len(dic)
		print "load vocab_word"
		print dic
		return dic
