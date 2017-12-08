
class Config(object):
	
	
	POSSIBLE_LABELS = 'silence unknown yes no up down left right on off stop go'.split()
	
	id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
	name2id = {name: i for i, name in id2name.items()}
	CLS_NUM = len(POSSIBLE_LABELS)
	
	non_flg_index = 0
	unk_flg_index = 1
	non_flg = 'silence'
	sil_flg = 'silence'
	unk_flg = 'unknown'
	
	sampling_rate = 16000
	
	
	@staticmethod
	def i2n(i):
		return Config.id2name.get(i, Config.non_flg_index)
	
	@staticmethod
	def n2i(n):
		return Config.name2id.get(n, Config.unk_flg)
