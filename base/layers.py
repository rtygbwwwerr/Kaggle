import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.seq2seq import AttentionWrapper, AttentionWrapperState, \
									   BasicDecoder, BeamSearchDecoder, dynamic_decode, \
									   TrainingHelper, ScheduledEmbeddingTrainingHelper, sequence_loss, tile_batch, \
									   BahdanauAttention, LuongAttention

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops

from abc import ABCMeta, abstractmethod				   
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
		
class ScheduledEmbeddingTrainingHelper_p(TrainingHelper):
	"""A training helper that adds scheduled sampling.
	
	Returns -1s for sample_ids where no sampling took place; valid sample id
	values elsewhere.
	"""

	def __init__(self, inputs, sequence_length, embedding, sampling_probability,
               time_major=False, seed=None, scheduling_seed=None, name=None):
		"""Initializer.
		
		Args:
		  inputs: A (structure of) input tensors.
		  sequence_length: An int32 vector tensor.
		  embedding: A callable that takes a vector tensor of `ids` (argmax ids),
		    or the `params` argument for `embedding_lookup`.
		  sampling_probability: A 0D `float32` tensor: the probability of sampling
		    categorically from the output ids instead of reading directly from the
		    inputs.
		  time_major: Python bool.  Whether the tensors in `inputs` are time major.
		    If `False` (default), they are assumed to be batch major.
		  seed: The sampling seed.
		  scheduling_seed: The schedule decision rule sampling seed.
		  name: Name scope for any created operations.
		
		Raises:
		  ValueError: if `sampling_probability` is not a scalar or vector.
		"""
# 		self.select_sample_val = -1
# 		self.next_inputs_val = -1
# 		self.base_next_inputs_val = -1
		with ops.name_scope(name, "ScheduledEmbeddingSamplingWrapper",
		                    [embedding, sampling_probability]):
			if callable(embedding):
				self._embedding_fn = embedding
			else:
				self._embedding_fn = (
			      lambda ids: embedding_ops.embedding_lookup(embedding, ids))
			self._sampling_probability = ops.convert_to_tensor(
			    sampling_probability, name="sampling_probability")
			if self._sampling_probability.get_shape().ndims not in (0, 1):
				raise ValueError(
				    "sampling_probability must be either a scalar or a vector. "
				    "saw shape: %s" % (self._sampling_probability.get_shape()))

			
			self._seed = seed
			self._scheduling_seed = scheduling_seed
			super(ScheduledEmbeddingTrainingHelper_p, self).__init__(
			    inputs=inputs,
			    sequence_length=sequence_length,
			    time_major=time_major,
			    name=name)
	
	def initialize(self, name=None):
		return super(ScheduledEmbeddingTrainingHelper_p, self).initialize(name=name)
	
	def sample(self, time, outputs, state, name=None):
		with ops.name_scope(name, "ScheduledEmbeddingTrainingHelperSample",
		                    [time, outputs, state]):
			# Return -1s where we did not sample, and sample_ids elsewhere
			select_sampler = bernoulli.Bernoulli(
			    probs=self._sampling_probability, dtype=dtypes.bool)
			select_sample = select_sampler.sample(
			    sample_shape=self.batch_size, seed=self._scheduling_seed)
			
# 			self.logs = tf.Print(select_sample, [select_sample])
# 			sample_id_sampler = categorical.Categorical(logits=outputs)
			sample_ids = math_ops.cast(math_ops.argmax(outputs, axis=-1), dtypes.int32)
# 			select_sample = tf.ones(shape=(self.batch_size,), dtype=dtypes.bool, name="test")
			return array_ops.where(
			    select_sample,
			    sample_ids,
			    gen_array_ops.fill([self.batch_size], -1))
# 			return sample_ids
	
	def next_inputs(self, time, outputs, state, sample_ids, name=None):
		with ops.name_scope(name, "ScheduledEmbeddingTrainingHelperNextInputs",
		                    [time, outputs, state, sample_ids]):
			(finished, base_next_inputs, state) = (
			    super(ScheduledEmbeddingTrainingHelper_p, self).next_inputs(
			        time=time,
			        outputs=outputs,
			        state=state,
			        sample_ids=sample_ids,
			        name=name))
			
			def maybe_sample():
				"""Perform scheduled sampling."""
				where_sampling = math_ops.cast(
				    array_ops.where(sample_ids > -1), dtypes.int32)
				where_not_sampling = math_ops.cast(
				    array_ops.where(sample_ids <= -1), dtypes.int32)
				sample_ids_sampling = array_ops.gather_nd(sample_ids, where_sampling)
				
				
				inputs_not_sampling = array_ops.gather_nd(
				    base_next_inputs, where_not_sampling)
				sampled_next_inputs = self._embedding_fn(sample_ids_sampling)
				base_shape = array_ops.shape(base_next_inputs)
				return (array_ops.scatter_nd(indices=where_sampling,
				                             updates=sampled_next_inputs,
				                             shape=base_shape)
				        + array_ops.scatter_nd(indices=where_not_sampling,
				                               updates=inputs_not_sampling,
				                               shape=base_shape))
			
			all_finished = math_ops.reduce_all(finished)
			next_inputs = control_flow_ops.cond(
			    all_finished, lambda: base_next_inputs, maybe_sample)
# 			self.next_inputs_val = next_inputs
# 			self.base_next_inputs_val = base_next_inputs
			return (finished, next_inputs, state)
		

class Seq2SeqBase(object):
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def _build_network_output(self):pass
	
	@abstractmethod
	def _build_loss(self, logits, targets):pass
	
	@abstractmethod	
	def _build_train_step(self, cost):pass

	@abstractmethod
	def _build_summary(self):pass
	
	
	def __init__(self):
		self._inputs = {}
		self._dict_ops = {}
		self._add_op(tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False), "global_step")
	def _add_op(self, op, op_name):
		self._dict_ops[op_name] = op
		
	def _build_inputs(self, num_features, sparse_x=False, sparse_y=False, dtype_x=tf.float32, dtype_y=tf.int32):
# 		# Has size [batch_size, max_stepsize, num_features], but the
# 		# batch_size and max_stepsize can vary along each step
# 		inputs = tf.placeholder(tf.float32, [None, None, num_features])
# 		
# 		# Here we use sparse_placeholder that will generate a
# 		# SparseTensor required by ctc_loss op.
# 		targets = tf.sparse_placeholder(tf.int32)
# 		
# 		# 1d array of size [batch_size]
# 		input_lenghts = tf.placeholder(tf.int32, [None])
		

		
		if sparse_x:
			self._inputs['X'] = tf.sparse_placeholder(dtype_x)
		else:
			self._inputs['X'] = tf.placeholder(
				shape=(None, None, num_features), # batch_size, max_time, feature_dim
				dtype=dtype_x,
				name='X'
			)
			
		if sparse_y:
			self._inputs['Y'] = tf.sparse_placeholder(dtype_y)
		else:
			self._inputs['Y'] = tf.placeholder(
				shape=(None, None), # batch_size, max_time, feature_dim
				dtype=dtype_y,
				name='Y'
			)
		
		self._inputs['X_lenghts'] = tf.placeholder(tf.int32, [None])
		
		self._inputs['Y_lenghts'] = tf.placeholder(tf.int32, [None])
		
		self._inputs['init_lr_rate'] = tf.placeholder(
			dtype=tf.float32,
			name='init_lr_rate'
		)

	
	
	def _make_feed_dict(self, data_dict):
		feed_dict = {}
		for key in data_dict.keys():
			
			if key in self._inputs:
				feed_dict[self._inputs[key]] = data_dict[key]
			else:pass
# 				print 'Unexpected argument {} in input dictionary!'.format(key)
		return feed_dict

	def _build_accuracy(self, decoded, targets):
		
		if isinstance(targets, tf.SparseTensor):
			targets = tf.sparse_tensor_to_dense(targets, default_value=-1)
			
		cut_len = tf.minimum(tf.shape(decoded)[1], tf.shape(targets)[1])
		decoder_lengths = self.get_input('Y_lenghts')
		
		with tf.variable_scope('accuracy_target'):
			flg = tf.equal(decoded[:, 0:cut_len],  targets[:, 0:cut_len])
			flg = tf.cast(flg, dtype=tf.float32)
			total_corrected = tf.reduce_sum(flg)
			acc = tf.divide(total_corrected, tf.cast(tf.reduce_sum(decoder_lengths), dtype=tf.float32), name='acc')
			
			flg_s = tf.reduce_sum(flg, axis=1)
			flg_y = tf.equal(tf.cast(flg_s, tf.int32), decoder_lengths)
			corrected_y = tf.reduce_sum(tf.cast(flg_y, dtype=tf.float32))
			seq_acc = tf.divide(corrected_y, tf.cast(tf.shape(decoded)[0], dtype=tf.float32), name='seq_acc')
		return 	acc, seq_acc
	
# 	def get_ops(self):
# 		'''
# 		return:ops, at least consist with 7 items:0:"train_op", 1:"summary", 2:"global_step", 3:"output", 4:"loss", 5:"acc", 6:"seq_acc"
# 		       you can add your own options after them.
# 		note:do not change the default order and names of the first seven items.
# 		'''
# 		ops = [self.train_op, self.summary, self.global_step, self.output, self.loss, self.accuracy, self.seq_accuracy]
# 		names = [None, None, None, "global_step", "loss", "acc", "seq_acc"]
# 		
# 		return ops, names
	
	def run_ops(self, sess, data_dict, names=None, exclude_names=None):
		
		feed_dict = self._make_feed_dict(data_dict)
		
		ops = None
		if names != None and len(names) > 0:
			ops = map(lambda name:self.get_op(name), names)
		else:
			names = self._dict_ops.keys()
			ops = self._dict_ops.values()
			
		
		
		if exclude_names is not None and len(exclude_names) > 0:
			for e_name in exclude_names:
				index = names.index(e_name)
				names.pop(index)
				ops.pop(index)
		
# 		print feed_dict
		values = sess.run(ops, feed_dict)
		
		dict_values = {k:v for k, v in zip(names, values)}
		return dict_values
	
	def get_op(self, name):
		return self._dict_ops.get(name)
	
	def get_input(self, name):
		return self._inputs[name]	
	
	def set_input(self, name, value):
		self._inputs[name] = value
	
	def _build_graph(self, num_features):
		self._build_inputs(num_features)
		targets = self.get_input('Y')
		

		logits, output, search_output = self._build_network_output()
		
		self._add_op(output, "output")
		
		self._add_op(search_output, "search_output")
		
		self._add_op(self._build_loss(logits, targets), "loss")
		
		self._add_op(self._build_train_step(self.get_op("loss")), "train_op")
		
		accuracy, seq_accuracy = self._build_accuracy(self.get_op("output"), targets)
		self._add_op(accuracy, "acc")
		self._add_op(seq_accuracy, "seq_acc")

		
		self._add_op(self._build_summary(), "summary")
		
		
		
		
		
		
		
