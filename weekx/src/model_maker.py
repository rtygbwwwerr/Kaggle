from __future__ import absolute_import
from recurrentshop import LSTMCell, RecurrentSequential
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Input
import recurrentshop
from recurrentshop.cells import *
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Activation
from keras.layers import add, multiply, concatenate
from keras import backend as K
from config import Config as cfg
from tensorflow.python.layers import core as layers_core
import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.seq2seq import AttentionWrapper, AttentionWrapperState, \
									   BasicDecoder, BeamSearchDecoder, dynamic_decode, \
									   TrainingHelper, sequence_loss, tile_batch, \
									   BahdanauAttention, LuongAttention
from nbformat.v4.tests.nbexamples import cells

									   
class CopyNetTrainingHelper(seq2seq.TrainingHelper):
	"""A helper for use during training.  Only reads inputs.
	
	Returned sample_ids are the argmax of the RNN output logits.
	"""
	
	def __init__(self, inputs, encoder_inputs_ids, sequence_length, 
									time_major=False, name=None):
		"""Initializer.
		
		Args:
		  inputs: A (structure of) input tensors.
		  sequence_length: An int32 vector tensor.
		  time_major: Python bool. Whether the tensors in `inputs` are time major.
			If `False` (default), they are assumed to be batch major.
		  name: Name scope for any created operations.
		
		Raises:
		  ValueError: if `sequence_length` is not a 1D tensor.
		"""
		super(CopyNetTrainingHelper, self).__init__(inputs, sequence_length,
				time_major=time_major, name=name)
		self.encoder_inputs_ids = encoder_inputs_ids

class CopyNetDecoder(seq2seq.BasicDecoder):
	"""
	copynet decoder, refer to the paper Jiatao Gu, 2016, 
	'Incorporating Copying Mechanism in Sequence-to-Sequence Learninag'
	https://arxiv.org/abs/1603.06393
	"""
	def __init__(self, config, cell, helper, initial_state, 
									encoder_outputs, output_layer):
		"""Initialize CopyNetDecoder.
		"""
		if output_layer is None:
			raise ValueError("output_layer should not be None")
		assert isinstance(helper, CopyNetTrainingHelper)
		self.encoder_outputs = encoder_outputs
		encoder_hidden_size = self.encoder_outputs.shape[-1].value
		self.copy_weight = tf.get_variable('copy_weight', 
								[encoder_hidden_size, cell.output_size])
		self.config = config
		super(CopyNetDecoder, self).__init__(cell, helper, initial_state, 
									output_layer=output_layer)
		
	@property
	def output_size(self):
		# Return the cell output and the id
		return seq2seq.BasicDecoderOutput(
			rnn_output=self._rnn_output_size() + 
						tf.convert_to_tensor(self.config.encoder_max_seq_len),
			sample_id=tensor_shape.TensorShape([]))
	
	def shape(self, tensor):
		s = tensor.get_shape()
		return tuple([s[i].value for i in range(0, len(s))])

	def _mix(self, generate_scores, copy_scores):
		# TODO is this correct? should verify the following code.
		"""
		B is batch_size, V is vocab_size, L is length of every input_id
		print genreate_scores.shape	 --> (B, V)
		print copy_scores.shape		 --> (B, L)
		print self._helper.inputs_ids   --> (B, L)
		"""
		print generate_scores.shape
		print copy_scores.shape
		print self._helper.encoder_inputs_ids.shape
		# mask is (B, L, V)
		mask = tf.one_hot(self._helper.encoder_inputs_ids, self.config.vocab_size)
		
		# choice one, move generate_scores to copy_scores
		expanded_generate_scores = tf.expand_dims(generate_scores, 1) # (B,1,V)
		actual_copy_scores = copy_scores + tf.reduce_sum(
								mask * expanded_generate_scores, 2)
		actual_generate_scores = generate_scores - tf.reduce_sum(
								mask * expanded_generate_scores, 1)
		
		# choice two, move copy_scores to generate_scores
		'''
		expanded_copy_scores = tf.expand_dims(copy_scores, 2)
		acutual_generate_scores = generate_scores + tf.reduce_sum(
									mask * expanded_copy_scores, 1)
		acutual_copy_scores = copy_scores - tf.reduce_sum(
									mask * expanded_copy_scores, 2)
		'''
		
		mix_scores = tf.concat([actual_generate_scores, actual_copy_scores], 1)
		mix_scores = tf.nn.softmax(mix_scores, -1) # mix_scores is (B, V+L)
		
		# make sure mix_socres.shape is (B, V + encoder_max_seq_len)
		padding_size = self.config.encoder_max_seq_len - self.shape(copy_scores)[1]
		mix_scores = tf.pad(mix_scores, [[0, 0], [0, padding_size]])

		return mix_scores

	def step(self, time, inputs, state, name=None):
		"""Perform a decoding step.
	
		Args:
		time: scalar `int32` tensor.
		inputs: A (structure of) input tensors.
		state: A (structure of) state tensors and TensorArrays.
		name: Name scope for any created operations.
	
		Returns:
		`(outputs, next_state, next_inputs, finished)`.
		"""
		with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
			cell_outputs, cell_state = self._cell(inputs, state)
			generate_scores = self._output_layer(cell_outputs)
			
			expand_cell_outputs = tf.expand_dims(cell_outputs, 1)
			copy_scores = tf.tensordot(self.encoder_outputs, self.copy_weight, 1)
			copy_scores.set_shape([self.encoder_outputs.get_shape()[0], self.encoder_outputs.get_shape()[1], self.copy_weight.get_shape()[1]])
			copy_scores = tf.nn.tanh(copy_scores)
			copy_scores = tf.reduce_sum(copy_scores * expand_cell_outputs, 2)
			
			mix_scores = self._mix(generate_scores, copy_scores)
			
			sample_ids = self._helper.sample(
				time=time, outputs=mix_scores, state=cell_state)
			# sample_ids are not always valid.. TODO
			(finished, next_inputs, next_state) = self._helper.next_inputs(
				time=time,
				outputs=mix_scores,
				state=cell_state,
				sample_ids=sample_ids)
		outputs = seq2seq.BasicDecoderOutput(mix_scores, sample_ids)
		return (outputs, next_state, next_inputs, finished)

# Thanks to 'initializers_enhanced.py' of Project RNN Enhancement:
# https://github.com/nicolas-ivanov/Seq2Seq_Upgrade_TensorFlow/blob/master/rnn_enhancement/initializers_enhanced.py
def orthogonal_initializer(scale=1.0):
	def _initializer(shape, dtype=tf.float32, partition_info=None):
		if partition_info is not None:
			ValueError(
				"Do not know what to do with partition_info in BN_LSTMCell")
		flat_shape = (shape[0], np.prod(shape[1:]))
		a = np.random.normal(0.0, 1.0, flat_shape)
		u, _, v = np.linalg.svd(a, full_matrices=False)
		q = u if u.shape == flat_shape else v
		q = q.reshape(shape)
		return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)
	return _initializer


# Thanks to https://github.com/OlavHN/bnlstm
def batch_norm(inputs, name_scope, is_training, epsilon=1e-3, decay=0.99):
	with tf.variable_scope(name_scope):
		size = inputs.get_shape().as_list()[1]

		scale = tf.get_variable(
			'scale', [size], initializer=tf.constant_initializer(0.1))
		offset = tf.get_variable('offset', [size])

		population_mean = tf.get_variable(
			'population_mean', [size],
			initializer=tf.zeros_initializer(), trainable=False)
		population_var = tf.get_variable(
			'population_var', [size],
			initializer=tf.ones_initializer(), trainable=False)
		batch_mean, batch_var = tf.nn.moments(inputs, [0])

		# The following part is based on the implementation of :
		# https://github.com/cooijmanstim/recurrent-batch-normalization
		train_mean_op = tf.assign(
			population_mean,
			population_mean * decay + batch_mean * (1 - decay))
		train_var_op = tf.assign(
			population_var, population_var * decay + batch_var * (1 - decay))

		if is_training is True:
			with tf.control_dependencies([train_mean_op, train_var_op]):
				return tf.nn.batch_normalization(
					inputs, batch_mean, batch_var, offset, scale, epsilon)
		else:
			return tf.nn.batch_normalization(
				inputs, population_mean, population_var, offset, scale,
				epsilon)


class BN_LSTMCell(tf.nn.rnn_cell.RNNCell):
	"""LSTM cell with Recurrent Batch Normalization.
	This implementation is based on:
		 http://arxiv.org/abs/1603.09025
	This implementation is also based on:
		 https://github.com/OlavHN/bnlstm
		 https://github.com/nicolas-ivanov/Seq2Seq_Upgrade_TensorFlow
	"""

	def __init__(self, num_units, is_training,
				 use_peepholes=False, cell_clip=None,
				 initializer=orthogonal_initializer(),
				 num_proj=None, proj_clip=None,
				 forget_bias=1.0,
				 state_is_tuple=True,
				 activation=tf.tanh):
		"""Initialize the parameters for an LSTM cell.
		Args:
		  num_units: int, The number of units in the LSTM cell.
		  is_training: bool, set True when training.
		  use_peepholes: bool, set True to enable diagonal/peephole
			connections.
		  cell_clip: (optional) A float value, if provided the cell state
			is clipped by this value prior to the cell output activation.
		  initializer: (optional) The initializer to use for the weight
			matrices.
		  num_proj: (optional) int, The output dimensionality for
			the projection matrices.  If None, no projection is performed.
		  forget_bias: Biases of the forget gate are initialized by default
			to 1 in order to reduce the scale of forgetting at the beginning of
			the training.
		  state_is_tuple: If True, accepted and returned states are 2-tuples of
			the `c_state` and `m_state`.  If False, they are concatenated
			along the column axis.
		  activation: Activation function of the inner states.
		"""
		if not state_is_tuple:
			tf.logging.log_first_n(
				tf.logging.WARN,
				"%s: Using a concatenated state is slower and "
				" will soon be deprecated.  Use state_is_tuple=True.", 1, self)

		self.num_units = num_units
		self.is_training = is_training
		self.use_peepholes = use_peepholes
		self.cell_clip = cell_clip
		self.num_proj = num_proj
		self.proj_clip = proj_clip
		self.initializer = initializer
		self.forget_bias = forget_bias
		self._state_is_tuple = state_is_tuple
		self.activation = activation

		if num_proj:
			self._state_size = (
				LSTMStateTuple(num_units, num_proj)
				if state_is_tuple else num_units + num_proj)
			self._output_size = num_proj
		else:
			self._state_size = (
				LSTMStateTuple(num_units, num_units)
				if state_is_tuple else 2 * num_units)
			self._output_size = num_units

	@property
	def state_size(self):
		return self._state_size

	@property
	def output_size(self):
		return self._output_size

	def __call__(self, inputs, state, scope=None):

		num_proj = self.num_units if self.num_proj is None else self.num_proj

		if self._state_is_tuple:
			(c_prev, h_prev) = state
		else:
			c_prev = tf.slice(state, [0, 0], [-1, self.num_units])
			h_prev = tf.slice(state, [0, self.num_units], [-1, num_proj])

		dtype = inputs.dtype
		input_size = inputs.get_shape().with_rank(2)[1]

		with tf.variable_scope(scope or type(self).__name__):
			if input_size.value is None:
				raise ValueError(
					"Could not infer input size from inputs.get_shape()[-1]")

			W_xh = tf.get_variable(
				'W_xh',
				[input_size, 4 * self.num_units],
				initializer=self.initializer)
			W_hh = tf.get_variable(
				'W_hh',
				[num_proj, 4 * self.num_units],
				initializer=self.initializer)
			bias = tf.get_variable('B', [4 * self.num_units])

			xh = tf.matmul(inputs, W_xh)
			hh = tf.matmul(h_prev, W_hh)

			bn_xh = batch_norm(xh, 'xh', self.is_training)
			bn_hh = batch_norm(hh, 'hh', self.is_training)

			# i:input gate, j:new input, f:forget gate, o:output gate
			lstm_matrix = tf.nn.bias_add(tf.add(bn_xh, bn_hh), bias)
			i, j, f, o = tf.split(
				value=lstm_matrix, num_or_size_splits=4, axis=1)

			# Diagonal connections
			if self.use_peepholes:
				w_f_diag = tf.get_variable(
					"W_F_diag", shape=[self.num_units], dtype=dtype)
				w_i_diag = tf.get_variable(
					"W_I_diag", shape=[self.num_units], dtype=dtype)
				w_o_diag = tf.get_variable(
					"W_O_diag", shape=[self.num_units], dtype=dtype)

			if self.use_peepholes:
				c = c_prev * tf.sigmoid(f + self.forget_bias +
										w_f_diag * c_prev) + \
					tf.sigmoid(i + w_i_diag * c_prev) * self.activation(j)
			else:
				c = c_prev * tf.sigmoid(f + self.forget_bias) + \
					tf.sigmoid(i) * self.activation(j)

			if self.cell_clip is not None:
				c = tf.clip_by_value(c, -self.cell_clip, self.cell_clip)

			bn_c = batch_norm(c, 'cell', self.is_training)

			if self.use_peepholes:
				h = tf.sigmoid(o + w_o_diag * c) * self.activation(bn_c)
			else:
				h = tf.sigmoid(o) * self.activation(bn_c)

			if self.num_proj is not None:
				w_proj = tf.get_variable(
					"W_P", [self.num_units, num_proj], dtype=dtype)

				h = tf.matmul(h, w_proj)
				if self.proj_clip is not None:
					h = tf.clip_by_value(h, -self.proj_clip, self.proj_clip)

			new_state = (LSTMStateTuple(c, h)
						 if self._state_is_tuple else tf.concat(1, [c, h]))

			return h, new_state


class LSTMDecoderCell(ExtendedRNNCell):

	def __init__(self, hidden_dim=None, **kwargs):
		if hidden_dim:
			self.hidden_dim = hidden_dim
		else:
			self.hidden_dim = self.output_dim
		super(LSTMDecoderCell, self).__init__(**kwargs)

	def build_model(self, input_shape):
		hidden_dim = self.hidden_dim
		output_dim = self.output_dim
		
		x = Input(batch_shape=input_shape)
		h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
		c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
		
		W1 = Dense(hidden_dim * 4,
				   kernel_initializer=self.kernel_initializer,
				   kernel_regularizer=self.kernel_regularizer,
				   use_bias=False)
		W2 = Dense(output_dim,
				   kernel_initializer=self.kernel_initializer,
				   kernel_regularizer=self.kernel_regularizer,)
		U = Dense(hidden_dim * 4,
				  kernel_initializer=self.kernel_initializer,
				  kernel_regularizer=self.kernel_regularizer,)
		
		z = add([W1(x), U(h_tm1)])
		
		z0, z1, z2, z3 = get_slices(z, 4)
		i = Activation(self.recurrent_activation)(z0)
		f = Activation(self.recurrent_activation)(z1)
		c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
		o = Activation(self.recurrent_activation)(z3)
		h = multiply([o, Activation(self.activation)(c)])
		y = Activation(self.activation)(W2(h))
		
		return Model([x, h_tm1, c_tm1], [y, h, c])


class AttentionDecoderCell(ExtendedRNNCell):

	def __init__(self, hidden_dim=None, **kwargs):
		if hidden_dim:
			self.hidden_dim = hidden_dim
		else:
			self.hidden_dim = self.output_dim
		self.input_ndim = 3
		super(AttentionDecoderCell, self).__init__(**kwargs)


	def build_model(self, input_shape):

		input_dim = input_shape[-1]
		output_dim = self.output_dim
		input_length = input_shape[1]
		hidden_dim = self.hidden_dim
		
		x = Input(batch_shape=input_shape)
		h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
		c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
		
		W1 = Dense(hidden_dim * 4,
				   kernel_initializer=self.kernel_initializer,
				   kernel_regularizer=self.kernel_regularizer)
		W2 = Dense(output_dim,
				   kernel_initializer=self.kernel_initializer,
				   kernel_regularizer=self.kernel_regularizer)
		W3 = Dense(1,
				   kernel_initializer=self.kernel_initializer,
				   kernel_regularizer=self.kernel_regularizer)
		U = Dense(hidden_dim * 4,
				  kernel_initializer=self.kernel_initializer,
				  kernel_regularizer=self.kernel_regularizer)
		print x.shape
		print c_tm1.shape
		print input_length
		C = Lambda(lambda x: K.repeat(x, input_length), output_shape=(input_length, input_dim))(c_tm1)
		print C.shape
		_xC = concatenate([x, C])
		print _xC.shape
		_xC = Lambda(lambda x: K.reshape(x, (-1, input_dim + hidden_dim)), output_shape=(input_dim + hidden_dim,))(_xC)
		
		alpha = W3(_xC)
		alpha = Lambda(lambda x: K.reshape(x, (-1, input_length)), output_shape=(input_length,))(alpha)
		alpha = Activation('softmax')(alpha)
		
		_x = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 1)), output_shape=(input_dim,))([alpha, x])
		
		z = add([W1(_x), U(h_tm1)])
		
		z0, z1, z2, z3 = get_slices(z, 4)
		
		i = Activation(self.recurrent_activation)(z0)
		f = Activation(self.recurrent_activation)(z1)
		
		c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
		o = Activation(self.recurrent_activation)(z3)
		h = multiply([o, Activation(self.activation)(c)])
		y = Activation(self.activation)(W2(h))
		
		return Model([x, h_tm1, c_tm1], [y, h, c])


def make_Seq2Seq(output_dim, output_length, batch_input_shape=None,
			input_shape=None, batch_size=None, input_dim=None, input_length=None,
			hidden_dim=None, depth=1, broadcast_state=True, unroll=False,
			stateful=False, inner_broadcast_state=True, teacher_force=False,
			peek=False, dropout=0.):

	'''
	Seq2seq model based on [1] and [2].
	This model has the ability to transfer the encoder hidden state to the decoder's
	hidden state(specified by the broadcast_state argument). Also, in deep models
	(depth > 1), the hidden state is propogated throughout the LSTM stack(specified by
	the inner_broadcast_state argument. You can switch between [1] based model and [2]
	based model using the peek argument.(peek = True for [2], peek = False for [1]).
	When peek = True, the decoder gets a 'peek' at the context vector at every timestep.

	[1] based model:

			Encoder:
			X = Input sequence
			C = LSTM(X); The context vector

			Decoder:
	y(t) = LSTM(s(t-1), y(t-1)); Where s is the hidden state of the LSTM (h and c)
	y(0) = LSTM(s0, C); C is the context vector from the encoder.

	[2] based model:

			Encoder:
			X = Input sequence
			C = LSTM(X); The context vector

			Decoder:
	y(t) = LSTM(s(t-1), y(t-1), C)
	y(0) = LSTM(s0, C, C)
	Where s is the hidden state of the LSTM (h and c), and C is the context vector
	from the encoder.

	Arguments:

	output_dim : Required output dimension.
	hidden_dim : The dimension of the internal representations of the model.
	output_length : Length of the required output sequence.
	depth : Used to create a deep Seq2seq model. For example, if depth = 3,
					there will be 3 LSTMs on the enoding side and 3 LSTMs on the
					decoding side. You can also specify depth as a tuple. For example,
					if depth = (4, 5), 4 LSTMs will be added to the encoding side and
					5 LSTMs will be added to the decoding side.
	broadcast_state : Specifies whether the hidden state from encoder should be
									  transfered to the deocder.
	inner_broadcast_state : Specifies whether hidden states should be propogated
													throughout the LSTM stack in deep models.
	peek : Specifies if the decoder should be able to peek at the context vector
			   at every timestep.
	dropout : Dropout probability in between layers.


	'''

	if isinstance(depth, int):
		depth = (depth, depth)
	if batch_input_shape:
		shape = batch_input_shape
	elif input_shape:
		shape = (batch_size,) + input_shape
	elif input_dim:
		if input_length:
			shape = (batch_size,) + (input_length,) + (input_dim,)
		else:
			shape = (batch_size,) + (None,) + (input_dim,)
	else:
		# TODO Proper error message
		raise TypeError
	if hidden_dim is None:
		hidden_dim = output_dim
	
	encoder = RecurrentSequential(readout=True, state_sync=inner_broadcast_state,
								  unroll=unroll, stateful=stateful,
								  return_states=broadcast_state)
	for _ in range(depth[0]):
		encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], hidden_dim)))
		encoder.add(Dropout(dropout))
	
	dense1 = TimeDistributed(Dense(hidden_dim))
	dense1.supports_masking = True
	dense2 = Dense(output_dim)
	
	decoder = RecurrentSequential(readout='add' if peek else 'readout_only',
								  state_sync=inner_broadcast_state, decode=True,
								  output_length=output_length, unroll=unroll,
								  stateful=stateful, teacher_force=teacher_force)
	
	for _ in range(depth[1]):
		decoder.add(Dropout(dropout, batch_input_shape=(shape[0], output_dim)))
		decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim,
									batch_input_shape=(shape[0], output_dim)))
	
	_input = Input(batch_shape=shape)
	_input._keras_history[0].supports_masking = True
	encoded_seq = dense1(_input)
	encoded_seq = encoder(encoded_seq)
	if broadcast_state:
		assert type(encoded_seq) is list
		states = encoded_seq[-2:]
		encoded_seq = encoded_seq[0]
	else:
		states = None
	encoded_seq = dense2(encoded_seq)
	inputs = [_input]
	if teacher_force:
		truth_tensor = Input(batch_shape=(shape[0], output_length, output_dim))
		truth_tensor._keras_history[0].supports_masking = True
		inputs += [truth_tensor]
	
	
	decoded_seq = decoder(encoded_seq,
						  ground_truth=inputs[1] if teacher_force else None,
						  initial_readout=encoded_seq, initial_state=states)
	
	model = Model(inputs, decoded_seq)
	model.encoder = encoder
	model.decoder = decoder
	return model

def make_AttentionSeq2Seq(output_dim, output_length, batch_input_shape=None,
					 batch_size=None, input_shape=None, input_length=None,
					 input_dim=None, hidden_dim=None, depth=1,
					 bidirectional=True, unroll=False, stateful=False, dropout=0.0,):
	'''
	This is an attention Seq2seq model based on [3].
	Here, there is a soft allignment between the input and output sequence elements.
	A bidirection encoder is used by default. There is no hidden state transfer in this
	model.
	
	The  math:
	
			Encoder:
			X = Input Sequence of length m.
			H = Bidirection_LSTM(X); Note that here the LSTM has return_sequences = True,
			so H is a sequence of vectors of length m.
	
			Decoder:
	y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
	and v (called the context vector) is a weighted sum over H:
	
	v(i) =  sigma(j = 0 to m-1)  alpha(i, j) * H(j)
	
	The weight alpha[i, j] for each hj is computed as follows:
	energy = a(s(i-1), H(j))
	alpha = softmax(energy)
	Where a is a feed forward network.
	
	'''

	if isinstance(depth, int):
		depth = (depth, depth)
	if batch_input_shape:
		shape = batch_input_shape
	elif input_shape:
		shape = (batch_size,) + input_shape
	elif input_dim:
		if input_length:
			shape = (batch_size,) + (input_length,) + (input_dim,)
		else:
			shape = (batch_size,) + (None,) + (input_dim,)
	else:
		# TODO Proper error message
		raise TypeError
	if hidden_dim is None:
		hidden_dim = output_dim
	
	_input = Input(batch_shape=shape)
	_input._keras_history[0].supports_masking = True
	
	encoder = RecurrentSequential(unroll=unroll, stateful=stateful,
								  return_sequences=True)
	encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[2])))
	
	for _ in range(1, depth[0]):
# 		encoder.add(Dropout(dropout))
		encoder.add(LSTMCell(hidden_dim))
	
	if bidirectional:
		encoder = Bidirectional(encoder, merge_mode='sum')
		encoder.forward_layer.build(shape)
		encoder.backward_layer.build(shape)
		# patch
		encoder.layer = encoder.forward_layer
	
	encoded = encoder(_input)
	decoder = RecurrentSequential(decode=True, output_length=output_length,
								  unroll=unroll, stateful=stateful)
# 	decoder.add(Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))
	if depth[1] == 1:
		decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
	else:
		decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
		for _ in range(depth[1] - 2):
		# 			decoder.add(Dropout(dropout))
			decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
		# 		decoder.add(Dropout(dropout))
		decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
	
	inputs = [_input]
	decoded = decoder(encoded)
	model = Model(inputs, decoded)
	return model

def make_teaching_attentionSeq2Seq(output_dim, output_length, batch_input_shape=None,
					 batch_size=None, input_shape=None, input_length=None,
					 input_dim=None, hidden_dim=None, depth=1,
					 teacher_force=True, bidirectional=True, unroll=False, stateful=False, dropout=0.0,):
	
	'''
	This is an attention Seq2seq model based on [3].
	Here, there is a soft allignment between the input and output sequence elements.
	A bidirection encoder is used by default. There is no hidden state transfer in this
	model.

	The  math:

			Encoder:
			X = Input Sequence of length m.
			H = Bidirection_LSTM(X); Note that here the LSTM has return_sequences = True,
			so H is a sequence of vectors of length m.

			Decoder:
	y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
	and v (called the context vector) is a weighted sum over H:

	v(i) =  sigma(j = 0 to m-1)  alpha(i, j) * H(j)

	The weight alpha[i, j] for each hj is computed as follows:
	energy = a(s(i-1), H(j))
	alpha = softmax(energy)
	Where a is a feed forward network.

	'''

	if isinstance(depth, int):
		depth = (depth, depth)
	if batch_input_shape:
		shape = batch_input_shape
	elif input_shape:
		shape = (batch_size,) + input_shape
	elif input_dim:
		if input_length:
			shape = (batch_size,) + (input_length,) + (input_dim,)
		else:
			shape = (batch_size,) + (None,) + (input_dim,)
	else:
		# TODO Proper error message
		raise TypeError
	if hidden_dim is None:
		hidden_dim = output_dim

	_input = Input(batch_shape=shape)
	_input._keras_history[0].supports_masking = True

	encoder = RecurrentSequential(unroll=unroll, stateful=stateful,
								  return_sequences=True)
	encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[2])))

	for _ in range(1, depth[0]):
# 		encoder.add(Dropout(dropout))
		encoder.add(LSTMCell(hidden_dim))

	if bidirectional:
		encoder = Bidirectional(encoder, merge_mode='sum')
		encoder.forward_layer.build(shape)
		encoder.backward_layer.build(shape)
		# patch
		encoder.layer = encoder.forward_layer

	encoded = encoder(_input)
# 	print encoded.shape
	decoder = RecurrentSequential(decode=True, output_length=output_length,
								  unroll=unroll, stateful=stateful, return_states=True)
# 	decoder.add(Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))
	if depth[1] == 1:
		decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
	else:
		decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
		for _ in range(depth[1] - 2):
# 			decoder.add(Dropout(dropout))
			decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
# 		decoder.add(Dropout(dropout))
		decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
	
	inputs = [_input]
	if teacher_force:
		truth_tensor = Input(batch_shape=(shape[0], output_length, output_dim))
		truth_tensor._keras_history[0].supports_masking = True
		inputs += [truth_tensor]
	
# 	decoded = decoder(encoded)
	decoded = decoder(encoded,
					  ground_truth=inputs[1] if teacher_force else None,
					  initial_readout=encoded, initial_state=None)
	
	model = Model(inputs, decoded[0])
	return model



def replace_with_unk(inputs):
	# TODO, should be wrong here, need to replace less_equal with less
	condition = tf.less_equal(inputs, tf.convert_to_tensor(cfg.vocab_size))
	return tf.where(condition, inputs, 
						tf.ones_like(inputs) * cfg.w2i(cfg.unk_flg))

# # placeholder for inputs
# encoder_inputs = tf.placeholder(tf.int32, shape=(cfg.batch_size, None))
# decoder_inputs = tf.placeholder(tf.int32, shape=(cfg.batch_size, None))
# decoder_outputs = tf.placeholder(tf.int32, shape=(cfg.batch_size, None))
# 
# # placeholder for sequence lengths
# encoder_inputs_lengths = tf.placeholder(tf.int32, shape=(cfg.batch_size,))
# decoder_inputs_lengths = tf.placeholder(tf.int32, shape=(cfg.batch_size,))

def make_tf_copyNet(args_input, mode='train'):
	assert mode in {'train', 'eval', 'infer'}, 'invalid mode!'
	# embedding maxtrix
	embedding_matrix = tf.get_variable("embedding_matrix", [
		cfg.vocab_size, cfg.embedding_size])
	encoder_inputs_emb = tf.nn.embedding_lookup(embedding_matrix, 
										replace_with_unk(args_input['encoder_inputs']))
	decoder_inputs_emb = tf.nn.embedding_lookup(embedding_matrix, 
										replace_with_unk(args_input['decoder_inputs']))

	# encoder
	encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(cfg.encoder_hidden_size)
	encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
								encoder_cell,
								encoder_inputs_emb,
								sequence_length=args_input['encoder_inputs_lengths'],
								time_major=False,
								dtype=tf.float32)
# 	encoder_outputs.set_shape([encoder_outputs.get_shape()[0], cfg.max_input_len, encoder_outputs.get_shape()[2]])
	# attention wrapper for decoder_cell
	attention_mechanism = seq2seq.LuongAttention(
							cfg.decoder_hidden_size, 
							encoder_outputs, 
							memory_sequence_length=args_input['encoder_inputs_lengths'])
	decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(cfg.decoder_hidden_size)
	decoder_cell = seq2seq.AttentionWrapper(
						decoder_cell, 
						attention_mechanism, 
						attention_layer_size=cfg.encoder_hidden_size)
	decoder_initial_state = decoder_cell.zero_state(cfg.batch_size, 
												dtype=tf.float32)
	projection_layer = layers_core.Dense(cfg.vocab_size, use_bias=False)

	# decoder
	if mode == 'infer':
		helper = seq2seq.GreedyEmbeddingHelper(
					embedding_matrix, 
					tf.fill([cfg.batch_size], cfg.w2i(cfg.start_flg),
					cfg.w2i(cfg.end_flg)))
		decoder = seq2seq.BasicDecoder(decoder_cell,
												  helper,
												  decoder_initial_state,
												  output_layer=projection_layer)
		maximum_iterations = tf.round(tf.reduce_max(args_input['encoder_inputs_lengths']) * 2)
		final_outputs, final_state, seq_len = \
				seq2seq.dynamic_decode(decoder,
									maximum_iterations=maximum_iterations)
		translations = final_outputs.sample_id
		return translations

	# train or eval mode
	helper = CopyNetTrainingHelper(decoder_inputs_emb, args_input['encoder_inputs'],
											   args_input['decoder_inputs_lengths'], 
											   time_major=False)
	decoder = CopyNetDecoder(cfg, decoder_cell, helper,
											  decoder_initial_state,
											  encoder_outputs,
											  output_layer=projection_layer)
# 	helper = seq2seq.TrainingHelper(args_input['encoder_inputs'],
# 											   args_input['decoder_inputs_lengths'], 
# 											   time_major=False)
# 	decoder = seq2seq.BasicDecoder(decoder_cell, helper,
# 											  decoder_initial_state,
# 											  output_layer=projection_layer)
	final_outputs, final_state, seq_lens = \
						seq2seq.dynamic_decode(decoder)
	logits = final_outputs.rnn_output

	# loss
	crossent = tf.nn.softmax_cross_entropy_with_logits(
			labels=args_input['decoder_outputs'], logits=logits)
	max_seq_len = logits.shape[1].value
	target_weights = tf.sequence_mask(args_input['decoder_inputs_lengths'], max_seq_len, 
										dtype=tf.float32)
	loss = tf.reduce_sum(crossent * target_weights / tf.to_float(
		cfg.batch_size))

	if mode == 'eval':
		return loss

	# gradient clip
	params = tf.trainable_variables()
	gradients = tf.gradients(loss, params)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients,
												  cfg.max_grad_norm)
	optimizer = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate)
	train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

	return loss, train_op

def variable_summaries(name, var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope(name):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

def make_tf_seq2seq_attention(args_input, mode='train'):
	assert mode in {'train', 'eval', 'infer'}, 'invalid mode!'
	# embedding maxtrix
# 	embedding_matrix = tf.get_variable("embedding_matrix", [
# 		cfg.vocab_size, cfg.embedding_size])
# 	encoder_inputs_emb = tf.nn.embedding_lookup(embedding_matrix, 
# 										replace_with_unk(args_input['encoder_inputs']))
# 	decoder_inputs_emb = tf.nn.embedding_lookup(embedding_matrix, 
# 										replace_with_unk(args_input['decoder_inputs']))
	
	# https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
	embedding_matrix = tf.Variable(tf.constant(0.0, shape=(cfg.vocab_size, cfg.embedding_size)),
								   trainable=True,   # if pre-trained
								   name="embedding_matrix")
	
	embedding_init = embedding_matrix.assign(args_input['embedding_placeholder'])
	x_embedding = tf.nn.embedding_lookup(embedding_matrix, args_input['encoder_inputs'])
	y_embedding = tf.nn.embedding_lookup(embedding_matrix, args_input['decoder_inputs'])
	
	# encoder
	cells = [tf.nn.rnn_cell.LSTMCell(cfg.encoder_hidden_size) for _ in range(cfg.n_encoder_layer)]
	encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
	encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
		cell=encoder_cell,
		dtype=tf.float32,
		inputs=x_embedding,
# 		sequence_length=args_input['encoder_inputs_lengths'],
	)
	
	variable_summaries('encoder_outputs', encoder_outputs)
	variable_summaries('encoder_final_state', encoder_final_state)
	print(encoder_outputs, encoder_final_state)
	

	cells = [tf.nn.rnn_cell.LSTMCell(cfg.decoder_hidden_size) for _ in range(cfg.n_decoder_layer)]
	# cells = [tf.nn.rnn_cell.DeviceWrapper(
	#	 tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.LSTMCell(latent_size)),
	#	 device='/gpu:%d' % i) for i in range(cfg.n_gpus)]
	
	decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
		cells=cells
	)
	
	attention_mechanism = tf.contrib.seq2seq.LuongAttention(
		num_units=cfg.encoder_hidden_size,
		memory=encoder_outputs
	)
	
	attention_cell = tf.contrib.seq2seq.AttentionWrapper(
		cell=decoder_cell,
		attention_mechanism=attention_mechanism,
		attention_layer_size=cfg.encoder_hidden_size  # optional
	)
	
	attention_zero_state = attention_cell.zero_state(
		batch_size=args_input['batch_size'], 
		dtype=tf.float32
	)
	attention_initial_state = attention_zero_state.clone(
		cell_state=encoder_final_state
	)
	
	training_helper = tf.contrib.seq2seq.TrainingHelper(
		inputs=y_embedding,
		sequence_length=args_input['decoder_outputs_lengths']
	)
	projection_layer = layers_core.Dense(cfg.vocab_size,
									   activation=tf.nn.sigmoid)
	
	decoder = tf.contrib.seq2seq.BasicDecoder(
		cell=attention_cell,
		helper=training_helper,
		initial_state=attention_initial_state,
		output_layer=projection_layer
	)
	
	final_outputs, final_state, final_sequence_lengths = \
		tf.contrib.seq2seq.dynamic_decode(
			decoder=decoder
		)
	variable_summaries('final_rnn_outputs', final_outputs.rnn_output)
	variable_summaries('final_cell_state', final_state.cell_state)
	variable_summaries('final_attention', final_state.attention)
	variable_summaries('final_alignments', final_state.alignments)
	

# 
	# decoder
	if mode == 'infer':
# 		helper = seq2seq.GreedyEmbeddingHelper(
# 					embedding_matrix, 
# 					tf.fill([cfg.batch_size], cfg.w2i(cfg.start_flg),
# 					cfg.w2i(cfg.end_flg)))
		helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
			embedding=args_input['embedding'],
			start_tokens=tf.constant(cfg.start_flg_index, shape=(args_input['batch_size'], )),
			end_token=cfg.end_flg_index
		)
		
		decoder = tf.contrib.seq2seq.BasicDecoder(
			cell=attention_cell,
			helper=helper,
			initial_state=attention_initial_state,
			output_layer=projection_layer
		)

# 		maximum_iterations = tf.round(tf.reduce_max(args_input['encoder_inputs_lengths']) * 2)
		final_outputs, final_state, seq_len = \
				seq2seq.dynamic_decode(decoder,
# 									maximum_iterations=cfg.max_input_len,
									)
		translations = final_outputs.sample_id
		return translations
# 
# 	# train or eval mode
# 	helper = seq2seq.TrainingHelper(args_input['decoder_inputs'],
# # 											   args_input['decoder_inputs_lengths'], 
# 											   time_major=False)
# 	
# 	decoder = seq2seq.BasicDecoder(decoder_cell, helper,
# 											  attention_initial_state,
# 											  output_layer=projection_layer)
# 	
# 	final_outputs, final_state, seq_lens = \
# 						seq2seq.dynamic_decode(decoder)

	logits = final_outputs.rnn_output # float32 [batch_size, sequence_length, num_decoder_symbols]
	
	targets = args_input['decoder_outputs']  # int32 [batch_size, sequence_length]
	weights = tf.cast(
		tf.sequence_mask(args_input['decoder_outputs_lengths'], maxlen=cfg.max_output_len), 
		tf.float32
	)  # float32 [batch_size, sequence_length]
	
	
# 	loss = tf.contrib.seq2seq.sequence_loss(
# 		logits, 
# 		targets, 
# 		weights,
# 		average_across_timesteps=True,
# 		average_across_batch=True
# 	)
# 	variable_summaries('loss', loss)


	# loss
	crossent = tf.nn.softmax_cross_entropy_with_logits(
			labels=targets, logits=logits)

	loss = tf.reduce_sum(crossent * weights / tf.to_float(
		cfg.batch_size))



	if mode == 'eval':
		return loss

	# gradient clip
	params = tf.trainable_variables()
	gradients = tf.gradients(loss, params)
# 	gradients, _ = tf.clip_by_global_norm(gradients,
# 												  cfg.max_grad_norm)
	optimizer = tf.train.AdamOptimizer()
# 	optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
	train_op = optimizer.apply_gradients(zip(gradients, params))

	return loss, train_op, embedding_init


def make_tf_tailored_seq2seq(
				encoder_hidden_size, 
				decoder_hidden_size, 
				batch_size, 
				embedding_dim, 
				vocab_size, 
				max_decode_iter_size,
				is_training = True,
				dropout = 0.9,
				encoder_cell_type = 'BN_LSTM',
				decoder_cell_type = 'BN_LSTM',
				n_encoder_layers = 1,
				n_decoder_layers = 1,
				attention_type = 'Bahdanau',
				attention_num_units=100,
				init_learning_rate=0.1,
				minimum_learning_rate=1e-5,
				decay_steps=1e4,
				decay_factor=0.3,
				attention_depth=100,
				is_bidrection=True,
				is_attention=True,
				beam_width=5,
				PAD=0, 
				START=1, 
				EOS=2,
				
		):
	return Seq2SeqModel(
				encoder_hidden_size = encoder_hidden_size, 
				decoder_hidden_size = decoder_hidden_size, 
				batch_size = batch_size, 
				embedding_dim = embedding_dim, 
				vocab_size = vocab_size, 
				is_training = is_training,
				dropout = dropout,
				encoder_cell_type = encoder_cell_type,
				decoder_cell_type = decoder_cell_type,
				n_encoder_layers = n_encoder_layers,
				n_decoder_layers = n_decoder_layers,
				attention_type = attention_type,
				attention_num_units = attention_num_units,
				max_decode_iter_size = max_decode_iter_size,
				init_learning_rate = init_learning_rate,
				minimum_learning_rate = minimum_learning_rate,
				decay_steps = decay_steps,
				decay_factor = decay_factor,
				attention_depth = attention_depth,
				is_bidrection = is_bidrection,
				is_attention = is_attention,
				beam_width = beam_width,
				PAD = PAD, 
				EOS = EOS, 
				START = START,
				input_batch=None)
	
	
class Seq2SeqModel:


	def __init__(self, 
				encoder_hidden_size, 
				decoder_hidden_size, 
				batch_size, 
				embedding_dim, 
				vocab_size, 
				is_training = True,
				dropout = 0.9,
				encoder_cell_type = 'BN_LSTM',
				decoder_cell_type = 'BN_LSTM',
				n_encoder_layers = 2,
				n_decoder_layers = 1,
				attention_type = 'Bahdanau',
				attention_num_units=100,
				max_decode_iter_size=10,
				init_learning_rate=0.01,
				minimum_learning_rate=1e-8,
				decay_steps=1e4,
				decay_factor=0.3,
				attention_depth=100,
				is_bidrection=True,
				is_attention=True,
				beam_width=5,
				PAD=0,
				START=1,
				EOS=2, 
				input_batch=None):
	
		self._is_training = is_training
		self._PAD = PAD
 		self._START = START
		self._EOS = EOS
		self._batch_size = None
		self._embedding_dim = embedding_dim
		self._vocab_size = vocab_size
		self._dropout = dropout
		self._n_encoder_layers = n_encoder_layers
		self._n_decoder_layers = n_decoder_layers
		self._encoder_hidden_size = encoder_hidden_size
		self._decoder_hidden_size = decoder_hidden_size
		self._encoder_cell_type = encoder_cell_type
		self._decoder_cell_type = decoder_cell_type
		self._is_bidrection = is_bidrection
		self._is_attention = is_attention
		self._beam_width = beam_width
		self._attention_type = attention_type
		self._attention_num_units = attention_num_units
		self._attention_depth = attention_depth
		self._max_decode_iter_size = max_decode_iter_size
		self._init_learning_rate = init_learning_rate
		self._decay_steps = decay_steps
		self._decay_factor = decay_factor
		self._minimum_learning_rate = minimum_learning_rate
		self._inputs = {}
		self._build_graph(input_batch)
# 	def _current_lr(self):
		
	def _build_train_step(self, loss):
		with tf.variable_scope('train'):
			train_step = tf.Variable(0, name='global_step', trainable=False)
			lr = tf.train.exponential_decay(
				self._init_learning_rate,
				train_step,
				self._decay_steps,
				self._decay_factor,
				staircase=True
			)
			lr = tf.clip_by_value(
				lr,
				self._minimum_learning_rate,
				self._init_learning_rate,
				name='lr_clip'
			)
			self.lr = lr
# 			opt = tf.train.GradientDescentOptimizer(self._lr)
			opt = tf.train.MomentumOptimizer(self.lr, 0.9)
# 			opt = tf.train.AdamOptimizer(learning_rate=self.lr)
			self._opt = opt
# 			opt = tf.train.AdagradOptimizer(learning_rate=0.01)
			self._grads = tf.constant(0.0)
			train_variables = tf.trainable_variables()
			grads_vars = opt.compute_gradients(loss, train_variables)
			
# 			tf.Print(grads_vars, [grads_vars], "grads=", summarize=10000)
			
			for i, (grad, var) in enumerate(grads_vars):
				grads_vars[i] = (tf.clip_by_norm(grad, 1.0), var)
				self._grads += tf.reduce_sum(tf.reduce_sum(grads_vars[i][0]), name="total_grads")
			apply_gradient_op = opt.apply_gradients(grads_vars, global_step=train_step)
			with tf.control_dependencies([apply_gradient_op]):
				train_op = tf.no_op(name='train_step')

		return train_step, train_op
	
	def _build_summary(self):
		tf.summary.scalar('learning_rate', self.lr)
		tf.summary.scalar('loss', self.loss)
		tf.summary.scalar('grads', self._grads)
		tf.summary.scalar('accuracy', self.accuracy)
		tf.summary.scalar('accuracy_seqs', self.accuracy_seqs)
		return tf.summary.merge_all()
	
	def _build_accuracy(self, decoder_result_ids, decoder_inputs, decoder_lengths):
		with tf.variable_scope('accuracy_target'):

			
			flg = tf.equal(decoder_result_ids, decoder_inputs[:, 0:tf.shape(decoder_result_ids)[1]])
			flg = tf.cast(flg, dtype=tf.float32)
			total_corrected = tf.reduce_sum(flg)
			acc = total_corrected / tf.cast(tf.reduce_sum(decoder_lengths), dtype=tf.float32)
			
			
			flg_s = tf.reduce_sum(flg, axis=1)
			flg_y = tf.equal(tf.cast(flg_s, tf.int32), decoder_lengths)
			corrected_y = tf.reduce_sum(tf.cast(flg_y, dtype=tf.float32))
			acc_whole = corrected_y / tf.cast(tf.shape(decoder_result_ids)[0], dtype=tf.float32)
# 			arr_pad = tf.zeros_like(decoder_inputs)
# 			arr_end = tf.ones_like(decoder_inputs) * self._EOS
# 			decoder_inputs_masked = tf.where(tf.equal(arr_pad, decoder_inputs), arr_end, decoder_inputs)
# 			
# 			flg_m = tf.equal(decoder_result_ids, decoder_inputs_masked[:, 0:tf.shape(decoder_result_ids)[1]])

		return acc, acc_whole
		
	def _build_loss(self, decoder_logits, decoder_inputs, decoder_lengths):
		with tf.variable_scope('loss_target'):
			#build decoder output, with appropriate padding and mask
			batch_size = tf.shape(decoder_inputs)[0]
			pads = tf.ones([batch_size, 1], dtype=tf.int32) * self._PAD
			paded_decoder_inputs = tf.concat([decoder_inputs, pads], 1)
			max_decoder_time = tf.reduce_max(decoder_lengths) + 1
			decoder_target = paded_decoder_inputs[:, :max_decoder_time]

			decoder_eos = tf.one_hot(decoder_lengths, depth=max_decoder_time,
									 on_value=1, off_value=self._PAD,
									 dtype=tf.int32)
			decoder_target += decoder_eos

			decoder_loss_mask = tf.sequence_mask(decoder_lengths + 1,
												 maxlen=max_decoder_time,
												 dtype=tf.float32)

		with tf.variable_scope('loss'):
			seq_loss = sequence_loss(
				decoder_logits,
				decoder_target,
				decoder_loss_mask,
				name='sequence_loss'
			)

		return seq_loss
	def _create_RNNLayers(self, state_size, dropout, n_encoder_layers, is_training):
		cells = None
		# build cell
		if self._encoder_cell_type == 'BN_LSTM':
			cells = BN_LSTMCell(state_size, is_training=is_training, forget_bias=1.0)
		elif self._encoder_cell_type == 'LSTM':
			cells = tf.nn.rnn_cell.LSTMCell(state_size, initializer=tf.orthogonal_initializer(), forget_bias=1.0)
# 				cells = [tf.nn.rnn_cell.LSTMCell(state_size) for _ in range(self._n_encoder_layers)]
# 				cells = tf.nn.rnn_cell.LSTMCell(state_size, initializer=tf.orthogonal_initializer())
		elif self._encoder_cell_type == 'GRU':
# 				cells = [tf.nn.rnn_cell.GRUCell(state_size) for _ in range(self._n_encoder_layers)]
			cells = tf.nn.rnn_cell.GRUCell(state_size, initializer=tf.orthogonal_initializer())

		else:
			raise ValueError
		if self._dropout > 0.0:
			cells = tf.nn.rnn_cell.DropoutWrapper(cell=cells, input_keep_prob=1.0, output_keep_prob=dropout)
		cells  = tf.nn.rnn_cell.MultiRNNCell([cells] * n_encoder_layers)
		return cells
	
	def _create_biRNNLayers(self, state_size, dropout, n_encoder_layers, is_training):
		fw_cells = None
		bw_cells = None
		# build cell
		if self._encoder_cell_type == 'BN_LSTM':
			fw_cells = BN_LSTMCell(state_size, is_training=is_training, forget_bias=1.0)
			bw_cells = BN_LSTMCell(state_size, is_training=is_training, forget_bias=1.0)
		elif self._encoder_cell_type == 'LSTM':
			fw_cells = tf.nn.rnn_cell.LSTMCell(state_size, initializer=tf.orthogonal_initializer(), forget_bias=1.0)
			bw_cells = tf.nn.rnn_cell.LSTMCell(state_size, initializer=tf.orthogonal_initializer(), forget_bias=1.0)
# 				cells = [tf.nn.rnn_cell.LSTMCell(state_size) for _ in range(self._n_encoder_layers)]
# 				cells = tf.nn.rnn_cell.LSTMCell(state_size, initializer=tf.orthogonal_initializer())
		elif self._encoder_cell_type == 'GRU':
# 				cells = [tf.nn.rnn_cell.GRUCell(state_size) for _ in range(self._n_encoder_layers)]
			fw_cells = tf.nn.rnn_cell.GRUCell(state_size, initializer=tf.orthogonal_initializer())
			bw_cells = tf.nn.rnn_cell.GRUCell(state_size, initializer=tf.orthogonal_initializer())
		else:
			raise ValueError
		if self._dropout > 0.0:
			fw_cells = tf.nn.rnn_cell.DropoutWrapper(cell=fw_cells, input_keep_prob=1.0, output_keep_prob=dropout)
			bw_cells = tf.nn.rnn_cell.DropoutWrapper(cell=bw_cells, input_keep_prob=1.0, output_keep_prob=dropout)
		fw_cells  = tf.nn.rnn_cell.MultiRNNCell([fw_cells] * n_encoder_layers)
		bw_cells  = tf.nn.rnn_cell.MultiRNNCell([bw_cells] * n_encoder_layers)
		return fw_cells, bw_cells
	
	def _build_encoder(self, encoder_inputs, encoder_lengths):
		with tf.variable_scope('word_embedding'):
			word_embedding = tf.get_variable(
				name="word_embedding",
				shape=(self._vocab_size, self._embedding_dim),
				initializer=xavier_initializer(),
				dtype=tf.float32
			)

			# batch_size, max_time, embed_dims
			encoder_input_vectors = tf.nn.embedding_lookup(word_embedding, encoder_inputs)

		with tf.variable_scope('encoder'):
			if self._is_bidrection:
				fw_cells, bw_cells = self._create_biRNNLayers(self._encoder_hidden_size, self._dropout, 
																self._n_encoder_layers, self._is_training)
				
				(fw_output, bw_output), (fw_final_state, bw_final_state) =\
					tf.nn.bidirectional_dynamic_rnn(
						fw_cells, bw_cells,
						encoder_input_vectors,
						sequence_length=encoder_lengths,
						time_major=False,
						dtype=tf.float32
					)

				encoder_outputs = tf.concat([fw_output, bw_output], 2)
				fw_final_state = fw_final_state[-1]
				bw_final_state = bw_final_state[-1]
				
				if isinstance(fw_final_state, LSTMStateTuple):
					encoder_state_c = tf.concat(
						[fw_final_state.c, bw_final_state.c], 1)
					encoder_state_h = tf.concat(
						[fw_final_state.h, bw_final_state.h], 1)
					encoder_final_state = LSTMStateTuple(encoder_state_c,
														 encoder_state_h)
				else:
					encoder_final_state = tf.concat(
						[fw_final_state, bw_final_state], 1)

			else:
				cells = self._create_RNNLayers(self._encoder_hidden_size, self._dropout, 
																self._n_encoder_layers, self._is_training)
				encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
					cells,
					encoder_input_vectors,
					sequence_length=encoder_lengths,
					time_major=False,
					dtype=tf.float32
				)
			return encoder_outputs, encoder_final_state
		
	def _build_decoder(self, encoder_outputs, encoder_final_state, encoder_lengths,
							 decoder_inputs, decoder_lengths):

		batch_size = tf.shape(decoder_inputs)[0]
		beam_width = self._beam_width
		tiled_batch_size = batch_size * beam_width

		with tf.variable_scope('decoder_cell'):
			state_size = self._decoder_hidden_size
			if self._is_bidrection:
				state_size = state_size * 2
			
			cells = None
			# build cell 
			#because decode layer is in front of the whole network, gradient vanish issue would not be serious
			#base on efficiency, we do not add batch normalizatioin
			if self._decoder_cell_type == 'LSTM' or self._decoder_cell_type == 'BN_LSTM':
				cells = tf.nn.rnn_cell.LSTMCell(state_size, initializer=tf.orthogonal_initializer(), forget_bias=0.0)
# 				cells = [tf.nn.rnn_cell.LSTMCell(state_size) for _ in range(self._n_decoder_layers)]
# 			elif self._decoder_cell_type == 'BN_LSTM':
# 				cells = BN_LSTMCell(state_size, is_training=True, forget_bias=1.0)
			elif self._decoder_cell_type == 'GRU':
# 				cells = [tf.nn.rnn_cell.GRUCell(state_size) for _ in range(self._n_decoder_layers)]
				cells = tf.nn.rnn_cell.GRUCell(state_size, initializer=tf.orthogonal_initializer())	
			else:
				raise ValueError
			if self._dropout > 0.0:
				cells = tf.nn.rnn_cell.DropoutWrapper(cell=cells, input_keep_prob=1.0, output_keep_prob=self._dropout)
#  			cells = tf.nn.rnn_cell.MultiRNNCell([cells] * self._n_decoder_layers)
			original_decoder_cell = cells

			with tf.variable_scope('beam_inputs'):
				tiled_encoder_outputs = tile_batch(encoder_outputs, beam_width)
				tiled_encoder_lengths = tile_batch(encoder_lengths, beam_width)

				if isinstance(encoder_final_state, LSTMStateTuple):
					tiled_encoder_final_state_c = tile_batch(encoder_final_state.c, beam_width)
					tiled_encoder_final_state_h = tile_batch(encoder_final_state.h, beam_width)
					tiled_encoder_final_state = LSTMStateTuple(tiled_encoder_final_state_c,
															   tiled_encoder_final_state_h)
				else:
					tiled_encoder_final_state = tile_batch(encoder_final_state, beam_width)

			if self._is_attention:
				attention_name = self._attention_type
				if attention_name == 'Bahdanau':
					attention_fn = BahdanauAttention
				elif attention_name == 'Luong':
					attention_fn = LuongAttention
				else:
					raise ValueError

				with tf.variable_scope('attention'):
					attention_mechanism = attention_fn(
						self._attention_num_units,
# 						normalize=True,
						encoder_outputs,
						encoder_lengths,
						name="attention_fn"
					)
					decoder_cell = AttentionWrapper(
						cells,
						attention_mechanism,
						attention_layer_size=self._attention_depth,
						output_attention=True,
					)
					decoder_initial_state = \
						decoder_cell.zero_state(batch_size, tf.float32).clone(
							cell_state=encoder_final_state
						)

				with tf.variable_scope('attention', reuse=True):
					beam_attention_mechanism = attention_fn(
						self._attention_num_units,
						tiled_encoder_outputs,
						tiled_encoder_lengths,
						name="attention_fn"
					)
					beam_decoder_cell = AttentionWrapper(
						original_decoder_cell,
						beam_attention_mechanism,
						attention_layer_size=self._attention_depth,
						output_attention=True
					)
					tiled_decoder_initial_state = \
						beam_decoder_cell.zero_state(tiled_batch_size, tf.float32).clone(
							cell_state=tiled_encoder_final_state
						)

			else:
				decoder_initial_state = encoder_final_state
				tiled_decoder_initial_state = decoder_cell.zero_state(tiled_batch_size, tf.float32)
				beam_decoder_cell = decoder_cell

		with tf.variable_scope('word_embedding', reuse=True):
			word_embedding = tf.get_variable(name="word_embedding")

		with tf.variable_scope('decoder'):
			out_func = layers_core.Dense(
				self._vocab_size, use_bias=False)

			goes = tf.ones([batch_size, 1], dtype=tf.int32) * self._START
			goes_decoder_inputs = tf.concat([goes, decoder_inputs], 1)
			embed_decoder_inputs = tf.nn.embedding_lookup(word_embedding, goes_decoder_inputs)

			training_helper = TrainingHelper(
				embed_decoder_inputs,
				decoder_lengths + 1
			)
			decoder = BasicDecoder(
				decoder_cell,
				training_helper,
				decoder_initial_state,
				output_layer=out_func,
			)

			decoder_outputs, decoder_state, decoder_sequence_lengths = \
				dynamic_decode(
					decoder,
					scope=tf.get_variable_scope(),
					maximum_iterations=self._max_decode_iter_size
				)

			tf.get_variable_scope().reuse_variables()

			start_tokens = tf.ones([batch_size], dtype=tf.int32) * self._START
			beam_decoder = BeamSearchDecoder(
				beam_decoder_cell,
				word_embedding,
				start_tokens,
				self._EOS,
				tiled_decoder_initial_state,
				beam_width,
				output_layer=out_func,
			)

			beam_decoder_outputs, beam_decoder_state, beam_decoder_sequence_lengths = \
				dynamic_decode(
					beam_decoder,
					scope=tf.get_variable_scope(),
					maximum_iterations=self._max_decode_iter_size
				)

		decoder_results = {
			'decoder_outputs': decoder_outputs[0],
			'decoder_result_ids': decoder_outputs[1],
			'decoder_state': decoder_state,
			'decoder_sequence_lengths': decoder_sequence_lengths,
			'beam_decoder_result_ids': beam_decoder_outputs.predicted_ids,
			'beam_decoder_scores': beam_decoder_outputs.beam_search_decoder_output.scores,
			'beam_decoder_state': beam_decoder_state,
			'beam_decoder_sequence_outputs': beam_decoder_sequence_lengths
		}
		return decoder_results
			
	def _build_inputs(self, input_batch):
		
# 		self._batch_size = tf.placeholder(shape=(), dtype=tf.int32, name='batch_size')
# 		batch_size = self._batch_size
		if input_batch is None:
			self._inputs['encoder_inputs'] = tf.placeholder(
				shape=(None, None), # batch_size, max_time
				dtype=tf.int32,
				name='encoder_inputs'
			)
			self._inputs['encoder_lengths'] = tf.placeholder(
				shape=(None,),
				dtype=tf.int32,
				name='encoder_lengths'
			)
			self._inputs['decoder_inputs'] = tf.placeholder(
				shape=(None, None), # batch_size, max_time
				dtype=tf.int32,
				name='decoder_inputs'
			)
			self._inputs['decoder_lengths'] = tf.placeholder(
				shape=(None,),
				dtype=tf.int32,
				name='decoder_lengths'
			)

		else:
			encoder_inputs, encoder_lengths, decoder_inputs, decoder_lengths = input_batch
			encoder_inputs.set_shape([None, None])
			decoder_inputs.set_shape([None, None])
			encoder_lengths.set_shape([None])
			decoder_lengths.set_shape([None])

			self._inputs = {
				'encoder_inputs': encoder_inputs,
				'encoder_lengths': encoder_lengths,
				'decoder_inputs': decoder_inputs,
				'decoder_lengths': decoder_lengths
			}

		return self._inputs['encoder_inputs'], self._inputs['encoder_lengths'], \
			   self._inputs['decoder_inputs'], self._inputs['decoder_lengths']

	def _build_graph(self, input_batch):
		encoder_inputs, encoder_lengths, decoder_inputs, decoder_lengths = self._build_inputs(input_batch)
		self.encoder_inputs = encoder_inputs

		encoder_outputs, encoder_state = self._build_encoder(encoder_inputs, encoder_lengths)
		decoder_result = self._build_decoder(encoder_outputs, encoder_state, encoder_lengths,
											 decoder_inputs, decoder_lengths)
		self.decoder_outputs = decoder_result['decoder_outputs']
		self.decoder_result_ids = decoder_result['decoder_result_ids']
		self.beam_search_result_ids = decoder_result['beam_decoder_result_ids']
		self.beam_search_scores = decoder_result['beam_decoder_scores']

		seq_loss = self._build_loss(self.decoder_outputs, decoder_inputs, decoder_lengths)
		self.train_step, self.train_op = self._build_train_step(seq_loss)
		self.loss = seq_loss
		self.accuracy, self.accuracy_seqs = self._build_accuracy(decoder_result['decoder_result_ids'], self._inputs['decoder_inputs'], self._inputs['decoder_lengths'])
		self.summary_op = self._build_summary()
	def make_feed_dict(self, data_dict):
		feed_dict = {}
		for key in data_dict.keys():
			try:
				feed_dict[self._inputs[key]] = data_dict[key]
			except KeyError:
				raise ValueError('Unexpected argument in input dictionary!')
		return feed_dict