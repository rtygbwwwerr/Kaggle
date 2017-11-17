from random import shuffle
import numpy as np


class SeqData:
	"""
	Data class interface for seq2seq model
	"""
	
	def __init__(self, PAD=0, EOS=1, UNK_flg='<UNK>'):
		"""
		Data format should be [(from_sequence, to_sequence), ....]
		"""
		self.max_length_x = None
		self.max_length_y = None
		self.symbols = None  # 0 always should be 'PAD', 1 always should be 'EOS'
		self._PAD = PAD
		self._UNK_flg = UNK_flg
		self._EOS = EOS
		self.train_sequences = list()
		self.val_sequences = list()

	@property
	def num_train_examples(self):
		return len(self.train_sequences)
	
	@property
	def num_val_examples(self):
		return len(self.val_sequences)
	
	@property
	def num_symbols(self):
		return len(self.symbols)
	
	@property
	def idx_to_symbol(self, symbol_idx):
		return self.symbols[symbol_idx]
	
	@property
	def initialized(self):
		return (self.max_length_x is not None) or (self.max_length_y is not None) or (self.num_symbols is not None)
	
	def _next_batch(self, data, batch_idxs):
		"""
		Generate next batch.
		:param data: data list to process
		:param batch_idxs: idxs to process
		:return: next data dict of batch_size amount data
		"""
		def _normalize_length(_data, max_length):
			return _data + [self._PAD] * (max_length - len(_data))
	
		def _empty_data():
			return _normalize_length([])
	
		from_data, from_lengths, to_data, to_lengths = [], [], [], []
		for idx in batch_idxs:
			from_sequence, to_sequence = data[idx]
			from_lengths.append(len(from_sequence))
			to_lengths.append(len(to_sequence))
			from_data.append(_normalize_length(from_sequence, self.max_length_x))
			to_data.append(_normalize_length(to_sequence, self.max_length_y))
	
		batch_data_dict = {
			'encoder_inputs': np.asarray(from_data, dtype=np.int32),
			'encoder_lengths': np.asarray(from_lengths, dtype=np.int32),
			'decoder_inputs': np.asarray(to_data, dtype=np.int32),
			'decoder_lengths': np.asarray(to_lengths, dtype=np.int32)
		}
		return batch_data_dict
	
	def _data_iterator(self, sequence, batch_size, random):
		idxs = list(range(len(sequence)))
		if random:
			shuffle(idxs)
	
		for start_idx in range(0, len(sequence), batch_size):
			end_idx = start_idx + batch_size
			next_batch = self._next_batch(sequence, idxs[start_idx:end_idx])
	
			# return batch only if the size of batch is original batch size
# 			if len(next_batch['encoder_inputs']) == batch_size:
			yield next_batch
	
	def train_datas(self, batch_size=16, random=True):
		"""
		Iterate through train data for single epoch
		:param batch_size: batch size
		:param random: if true, iterate randomly
		:return: train data iterator
		"""
		assert self.initialized, "Dataset is not initialized!"
		return self._data_iterator(self.train_sequences, batch_size, random)
	
	def val_datas(self, batch_size=16, random=True):
		"""
		Iterate through validaiton data for single epoch
		:param batch_size: batch size
		:param random: if true, iterate randomly
		:return: validation data iterator
		"""
		assert self.initialized, "Dataset is not initialized!"
		return self._data_iterator(self.val_sequences, batch_size, random)
	
	def train_data_by_idx(self, start, end):
		assert start >= 0 and end <= len(self.train_sequences)
		return self._next_batch(self.train_sequences, range(start, end))
	
	def val_data_by_idx(self, start, end):
		assert start >= 0 and end < len(self.val_sequences)
		return self._next_batch(self.val_sequences, range(start, end))
	
	def interpret(self, ids, join_string=''):
		real_ids = []
		for _id in ids:
			if _id != self._EOS:
				real_ids.append(_id)
			else:
				break
	
		return join_string.join(str(self.symbols.get(ri, self._UNK_flg)) for ri in real_ids)
	
	def interpret_result(self, input_ids, output_ids, infer_output_ids, show_num=3):
		for i in range(show_num):
			input_sequence = ' '.join(self.interpret(input_ids[i], join_string=' ')
			                          .replace(self.symbols[self._PAD], '').split())
			output_sequence = ' '.join(self.interpret(output_ids[i], join_string=' ')
			                           .replace(self.symbols[self._PAD], '').split())
			infer_output_sequence = self.interpret(infer_output_ids[i], join_string=' ')
			
			# temporary for calculation seq data
			print('{} -> {} (Real: {})'.format(input_sequence, infer_output_sequence, output_sequence))
	
	
	def eval_result(self, input_ids, output_ids, infer_output_ids, step, batch_size, print_detil=True):
		_right, _wrong = 0.0, 0.0
		
		infos = []
		for i in range(len(input_ids)):
			input_sequence = ' '.join(self.interpret(input_ids[i], join_string=' ')
			                          .replace(self.symbols[self._PAD], '').split())
			output_sequence = ' '.join(self.interpret(output_ids[i], join_string=' ')
			                           .replace(self.symbols[self._PAD], '').split())
			infer_output_sequence = self.interpret(infer_output_ids[i], join_string=' ')
			info = 'EVAL:{}==>{} -> {} (Real: {})'.format(step * batch_size + i, input_sequence, infer_output_sequence, output_sequence)
			try:
				if output_sequence.strip() == infer_output_sequence.strip():
					info = "{}:{}".format("[Right]", info)
					_right += 1.0
				else:
					info = "{}:{}".format("[False]", info)
					_wrong += 1.0
			except ValueError:  # output_sequence == ''
				_wrong += 1.0
			if print_detil:
				print info
			infos.append(info)
		return _right, _wrong, infos

	def build(self):
		"""
		Build data and save in self.train_sequences, self.val_sequences
		"""
		raise NotImplementedError
	
	def load(self):
		"""
		Load data and save in self.train_sequences, self.val_sequences
		"""
		raise NotImplementedError
	
class NumpySeqData(SeqData):
	
	@staticmethod
	def _convert_numpydata_to_list(x, y, PAD):
		from_lens = x.shape[1] - np.sum(x==PAD, axis=1)
		to_lens = y.shape[1] - np.sum(y==PAD, axis=1)
		#delete END flag
# 		to_lens = to_lens - 1
		print "max input len:%d, max output len is:%d"%(np.max(from_lens), np.max(to_lens))
		for i in range(len(from_lens)):
			yield [list(x[i, :from_lens[i]]), list(y[i, :to_lens[i]])]
			
	def load(self, X_train, Y_train, X_valid, Y_valid, vocab_i2w):
		self.train_sequences = list()
		self.val_sequences = list()
		self.symbols = vocab_i2w
		if X_train is not None:
			self.max_length_x = X_train.shape[1]
		else:
			self.max_length_x = X_valid.shape[1]
		if Y_train is not None:
			self.max_length_y = Y_train.shape[1]
		else:
			self.max_length_y = Y_valid.shape[1]
		self._train_x_array = X_train
		self._train_y_array = Y_train
		self._valid_x_array = X_valid
		self._valid_y_array = Y_valid

	def build(self):
		if self._train_x_array is not None:
			self.train_sequences = list(NumpySeqData._convert_numpydata_to_list(self._train_x_array, self._train_y_array, self._PAD))
		if self._valid_x_array is not None:
			self.val_sequences = list(NumpySeqData._convert_numpydata_to_list(self._valid_x_array, self._valid_y_array, self._PAD))
