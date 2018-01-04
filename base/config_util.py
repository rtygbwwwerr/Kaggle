

class Vocab:
	dic_i2w={}
	dic_w2i={}
	pad_flg = '<PAD>'
	unk_flg = '<UNK>'
	start_flg = '<GO>'
	end_flg = '<EOS>'
	size = 0
	
	@property
	def pad_flg_index(self):
		return self.dic_w2i[self.pad_flg]
	
	@property
	def unk_flg_index(self):
		return self.dic_w2i[self.unk_flg]
	
	@property
	def start_flg_index(self):
		return self.dic_w2i[self.start_flg]
	
	@property
	def end_flg_index(self):
		return self.dic_w2i[self.end_flg]
	
	
	def __init__(self, dic_w2i):
		self.dic_w2i = dic_w2i.copy()
		self.dic_i2w = {i: w for w, i in dic_w2i.items()}
		self.size = len(dic_w2i)
	def i2w(self, i):
		return self.dic_i2w.get(i, self.unk_flg)
	
	def w2i(self, w):
		return self.dic_w2i.get(w, self.unk_flg_index)
	
	def w_in(self, w):
		return w in self.dic_w2i
	
	def i_in(self, i):
		return i in self.dic_i2w
	
	def wordset(self):
		return set(self.dic_w2i.keys())
	
		
