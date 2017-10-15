import data_process

class Config(object):
	
	dic_output_word2i = {}
	dic_output_i2word ={}
	output_vocab_size = 0
	dic_constant = None
	
	@staticmethod
	def init():
		Config.dic_output_word2i = data_process.load_dict('../data/out_vocab.csv')
		Config.output_vocab_size = len(Config.dic_output_word2i)
		Config.dic_output_i2word = {v: k for k, v in Config.dic_output_word2i.items()}
		print 'len w2i:%d, len i2w:%d'%(len(Config.dic_output_word2i), len(Config.dic_output_i2word))
		Config.dic_constant = data_process.load_input_dict()
		print 'input constant word num:%d'%(len(Config.dic_constant))
		
	batch_size = 85
	max_num_features = 31
# 	max_input_len = 1120,550,280
	max_input_len = 140
# 	max_output_len = 1850,411
	max_output_len = 111
	max_token_char_num = 40
	word_vec_len = 255
	input_vocab_size = 258

	input_hidden_dim = 512
	space_letter = 0
# 	boundary_word = -3
	pad_size = 1
	boundary_start = 2
	boundary_end = 3
	
	start_flg = '<GO>'
	end_flg = '<EOS>'
	copy_flg = '<UNK>'
	unk_flg = '<UNK>'
	pad_flg = '<PAD>'
	non_flg = '<PAD>'
	non_flg_index = 0
	pad_flg_index = 0
	unk_flg_index = 3
	start_flg_index = 1
	end_flg_index = 2
	copy_flag_index = 3
	start_word = [boundary_start]*(word_vec_len-2)
	end_word = [boundary_end]*(word_vec_len-2)
	input_split = 1
	
	max_input_char_num = 1300
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