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
import tensorflow.contrib.slim as slim
import math
from abc import ABCMeta, abstractmethod

def ds_cnn_arg_scope(scale_l1, scale_l2):
	"""Defines the default ds_cnn argument scope.
	Args:
		weight_decay: The weight decay to use for regularizing the model.
	Returns:
		An `arg_scope` to use for the DS-CNN model.
	"""
	with slim.arg_scope(
			[slim.convolution2d, slim.separable_convolution2d],
			weights_initializer=slim.initializers.xavier_initializer(),
			biases_initializer=slim.init_ops.zeros_initializer(),
			weights_regularizer=slim.l1_l2_regularizer(scale_l1=scale_l1, scale_l2=scale_l2)) as sc:
		return sc

def depthwise_separable_conv2(inputs,
								num_pwc_filters,
								sc,
								kernel_size,
								w_scale_l1,
								w_scale_l2,
								b_scale_l1,
								b_scale_l2,				
								stride):
	""" Helper function to build the depth-wise separable convolution layer.
	"""

	# skip pointwise by setting num_outputs=None
	depthwise_conv = slim.separable_convolution2d(inputs,
													num_outputs=None,
													stride=stride,
# 													weights_regularizer=slim.l1_l2_regularizer(w_scale_l1, w_scale_l2),
# 													biases_regularizer=slim.l1_l2_regularizer(b_scale_l1, b_scale_l2),
													depth_multiplier=1,
													kernel_size=kernel_size,
													scope=sc+'/depthwise_conv')

	bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
	pointwise_conv = slim.convolution2d(bn,
										num_pwc_filters,
										kernel_size=[1, 1],
# 										weights_regularizer=slim.l1_l2_regularizer(w_scale_l1, w_scale_l2),
# 										biases_regularizer=slim.l1_l2_regularizer(b_scale_l1, b_scale_l2),
										scope=sc+'/pointwise_conv')
	
	bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
	return bn


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
			the projection matrices.	If None, no projection is performed.
			forget_bias: Biases of the forget gate are initialized by default
			to 1 in order to reduce the scale of forgetting at the beginning of
			the training.
			state_is_tuple: If True, accepted and returned states are 2-tuples of
			the `c_state` and `m_state`.	If False, they are concatenated
			along the column axis.
			activation: Activation function of the inner states.
		"""
		if not state_is_tuple:
			tf.logging.log_first_n(
				tf.logging.WARN,
				"%s: Using a concatenated state is slower and "
				" will soon be deprecated.	Use state_is_tuple=True.", 1, self)

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
			time_major: Python bool.	Whether the tensors in `inputs` are time major.
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
		
"""Base class for all models."""


import tensorflow as tf

OPTIMIZER_CLS_NAMES = {
	"adagrad": tf.train.AdagradOptimizer,
	"adadelta": tf.train.AdadeltaOptimizer,
	"adam": tf.train.AdamOptimizer,
	"rmsprop": tf.train.RMSPropOptimizer,
	"sgd": tf.train.GradientDescentOptimizer,
	"momentum": tf.train.MomentumOptimizer,
	"nestrov": tf.train.MomentumOptimizer
}


class ModelBase(object):
	
	__metaclass__ = ABCMeta
	def __init__(self):
		self._inputs = {}
		self._dict_ops = {}
		self._add_op(tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False), "global_step")
	def _add_op(self, op, op_name):
		self._dict_ops[op_name] = op
	def get_op(self, name):
		return self._dict_ops.get(name)
	
	def get_input(self, name):
		return self._inputs[name]	
	
	def set_input(self, name, value):
		self._inputs[name] = value
		
	def run_ops(self, sess, data_dict, names=None, exclude_names=None):
		
		feed_dict = self.make_null_feed_dict(data_dict)
		
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
	
	def _make_null_array(self, val):
		val_arr = None
		shape = val.shape
		type = val.dtype.as_numpy_dtype
		if shape is None or str(shape) == '<unknown>' or len(shape) == 0:
			val_arr = 0
		else:
			arr_shape = [1] * len(shape)
			val_arr = np.zeros(arr_shape, dtype=type)
		return val_arr
	
	def make_null_feed_dict(self, data_dict):
		feed_dict = self._make_feed_dict(data_dict)
		for key, val in self._inputs.iteritems():
			if key not in data_dict:
				feed_dict[self._inputs[key]] = self._make_null_array(val)
		return feed_dict
				
	def _make_feed_dict(self, data_dict):
		feed_dict = {}
		for key in data_dict.keys():
			
			if key in self._inputs:
				feed_dict[self._inputs[key]] = data_dict[key]
			else:pass
# 				print 'Unexpected argument {} in input dictionary!'.format(key)
		return feed_dict
	
	def _add_noise_to_inputs(self, inputs, stddev=0.075):
		"""Add gaussian noise to the inputs.
		Args:
			inputs: the noise free input-features.
			stddev (float, optional): The standart deviation of the noise.
				Default is 0.075.
		Returns:
			inputs: Input features plus noise.
		"""
		# if stddev != 0:
		#	 with tf.variable_scope("input_noise"):
		#		 # Add input noise with a standart deviation of stddev.
		#		 inputs = tf.random_normal(
		#			 tf.shape(inputs), 0.0, stddev) + inputs
		# return inputs
		raise NotImplementedError

	def _add_noise_to_gradients(self, grads_and_vars, gradient_noise_scale,
								stddev=0.075):
		"""Adds scaled noise from a 0-mean normal distribution to gradients.
		Args:
			grads_and_vars:
			gradient_noise_scale:
			stddev (float):
		Returns:
		"""
		raise NotImplementedError

	def _set_optimizer(self, optimizer, learning_rate):
		"""Set optimizer.
		Args:
			optimizer (string): the name of the optimizer in
				OPTIMIZER_CLS_NAMES
			learning_rate (float): A learning rate
		Returns:
			optimizer:
		"""
		optimizer = optimizer.lower()
		if optimizer not in OPTIMIZER_CLS_NAMES:
			raise ValueError(
				"Optimizer name should be one of [%s], you provided %s." %
				(", ".join(OPTIMIZER_CLS_NAMES), optimizer))

		# Select optimizer
		if optimizer == 'momentum':
			return OPTIMIZER_CLS_NAMES[optimizer](
				learning_rate=learning_rate,
				momentum=0.9)
		elif optimizer == 'nestrov':
			return OPTIMIZER_CLS_NAMES[optimizer](
				learning_rate=learning_rate,
				momentum=0.9,
				use_nesterov=True)
		else:
			return OPTIMIZER_CLS_NAMES[optimizer](
				learning_rate=learning_rate)

	def train(self, loss, optimizer, learning_rate):
		"""Operation for training. Only the sigle GPU training is supported.
		Args:
			loss: An operation for computing loss
			optimizer (string): name of the optimizer in OPTIMIZER_CLS_NAMES
			learning_rate (placeholder): A learning rate
		Returns:
			train_op: operation for training
		"""
		# Create a variable to track the global step
		global_step = tf.Variable(0, name='global_step', trainable=False)

		# Set optimizer
		self.optimizer = self._set_optimizer(optimizer, learning_rate)

		if self.clip_grad_norm is not None:
			# Compute gradients
			grads_and_vars = self.optimizer.compute_gradients(loss)

			# Clip gradients
			clipped_grads_and_vars = self._clip_gradients(grads_and_vars)

			# Create operation for gradient update
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				train_op = self.optimizer.apply_gradients(
					clipped_grads_and_vars,
					global_step=global_step)

		else:
			# Use the optimizer to apply the gradients that minimize the loss
			# and also increment the global step counter as a single training
			# step
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				train_op = self.optimizer.minimize(
					loss, global_step=global_step)

		return train_op

	def _clip_gradients(self, grads_and_vars):
		"""Clip gradients.
		Args:
			grads_and_vars (list): list of tuples of `(grads, vars)`
		Returns:
			clipped_grads_and_vars (list): list of tuple of
				`(clipped grads, vars)`
		"""
		# TODO: Optionally add gradient noise

		clipped_grads_and_vars = []

		# Clip gradient norm
		for grad, var in grads_and_vars:
			if grad is not None:
				clipped_grads_and_vars.append(
					(tf.clip_by_norm(grad, clip_norm=self.clip_grad_norm),
					 var))

		# Clip gradient
		# for grad, var in grads_and_vars:
		#	 if grad is not None:
		#		 clipped_grads_and_vars.append(
		#			 (tf.clip_by_value(grad,
		#								 clip_value_min=-self.clip_grad_norm,
		#								 clip_value_max=self.clip_grad_norm),
		#				var))

		# TODO: Add histograms for variables, gradients (norms)
		# self._tensorboard(trainable_vars)

		return clipped_grads_and_vars

	@abstractmethod
	def _build_network_output(self):pass
	
	@abstractmethod
	def _build_loss(self, logits, targets):pass
	
	@abstractmethod	
	def _build_train_step(self, cost):pass

	@abstractmethod
	def _build_summary(self):pass


	
class ClassifierBase(ModelBase):
	

	
	def __init__(self, input_info, model_info):
		ModelBase.__init__(self)
		self._model_info = model_info
		self._input_info = input_info
	
	@staticmethod
	def get_x_name(index):
		return 'X_{}'.format(index)
	
	def _build_inputs(self, dtype_x=tf.float32, dtype_y=tf.int32):
		x_dims = self._input_info['x_dims']
		for i, x_dim in enumerate(x_dims):
			self.set_input(ClassifierBase.get_x_name(i), tf.placeholder(
# 				shape=(None, None, x_dim[1]), # batch_size, max_time, feature_dim
				shape=(None,) + x_dim,
				dtype=dtype_x,
				name=ClassifierBase.get_x_name(i)
			))
		self.set_input('Y', tf.placeholder(
			shape=(None, None), # batch_size, max_time
			dtype=dtype_y,
			name='Y'
		))
		
		self.set_input('init_lr_rate', tf.placeholder(
			dtype=tf.float32,
			name='init_lr_rate'
		))
		self.set_input('dropout_prob', tf.placeholder(dtype=tf.float32, name='dropout_prob'))
		self.set_input('decay_step', tf.placeholder(dtype=tf.int32, name='decay_step'))
		self.set_input('decay_factor', tf.placeholder(dtype=tf.float32, name='decay_factor'))

		
	def _build_loss(self, logits, targets):
		with tf.name_scope('cross_entropy'):
# 			tf.nn.sampled_softmax_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true, sampled_values, remove_accidental_hits, partition_strategy, name)
			cp_loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
			
			weight = self._model_info.get('cls_weight', None)
			if weight is not None:
				weight = ops.convert_to_tensor(weight)
				targets = tf.cast(targets, tf.float32)
				cp_loss = tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=logits, pos_weight=weight)
			loss = tf.reduce_mean(cp_loss)
		return loss
	
	def _build_accuracy(self, logits, targets):
		predicted_indices = tf.argmax(logits, 1)
		expected_indices = tf.argmax(targets, 1)
		correct_prediction = tf.equal(predicted_indices, expected_indices)
		
		confusion_matrix = tf.confusion_matrix(
		    expected_indices, predicted_indices, num_classes=self._input_info['num_cls'])
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		return accuracy, confusion_matrix
	
	def _build_train_step(self, loss):
		with tf.variable_scope('train'):
			lr = tf.train.exponential_decay(
				self.get_input("init_lr_rate"),
				self.get_op('global_step'),
				self.get_input("decay_step"),
				self.get_input("decay_factor"),
				staircase=True
			)
			lr = tf.clip_by_value(
				lr,
				1e-5,
				self.get_input('init_lr_rate'),
				name='lr_clip'
			)
			opt = OPTIMIZER_CLS_NAMES.get(self._model_info['opt'], tf.train.AdadeltaOptimizer)(lr)
# 			opt = tf.train.GradientDescentOptimizer(self._lr)
# 			opt = tf.train.MomentumOptimizer(self.lr, 0.9)
# 			opt = tf.train.RMSPropOptimizer(learning_rate=lr)
# 			opt = tf.train.AdadeltaOptimizer(learning_rate=lr)
# 			opt = tf.train.AdamOptimizer(learning_rate=self.lr)
# 			opt = tf.train.AdagradOptimizer(learning_rate=0.01)
			grads = tf.constant(0.0)
			train_variables = tf.trainable_variables()
			grads_vars = opt.compute_gradients(loss, train_variables)
			
			self._add_op(lr, "lr")
			for i, (grad, var) in enumerate(grads_vars):
				grads_vars[i] = (tf.clip_by_norm(grad, 1.0), var)
				grads += tf.reduce_sum(tf.reduce_sum(grads_vars[i][0]), name="total_grads")

			apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.get_op('global_step'))
			with tf.control_dependencies([apply_gradient_op]):
				train_op = tf.no_op(name='train_step')
				
		return train_op
	
	def _build_summary(self):
		tf.summary.scalar('learning_rate', self.get_op('lr'))
		tf.summary.scalar('loss', self.get_op('loss'))
		tf.summary.scalar('accuracy', self.get_op('acc'))
		return tf.summary.merge_all()
	
	def _build_graph(self):
		self._build_inputs()
		targets = self.get_input('Y')
		

		logits, output = self._build_network_output()
		self._add_op(output, "output")
		self._add_op(output, "logits")
		
		self._add_op(self._build_loss(logits, targets), "loss")
		
		self._add_op(self._build_train_step(self.get_op("loss")), "train_op")
		
		accuracy, confuse_matrix = self._build_accuracy(logits, targets)
		self._add_op(accuracy, "acc")
		self._add_op(confuse_matrix, "cf_mat")

		self._add_op(self._build_summary(), "summary")
				
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
	def _make_null_array(self, shape, type):
		val_arr = None
		if shape is None or len(shape) == 0:
			val_arr = 0
		else:
			arr_shape = [1] * len(shape)
			val_arr = np.zeros(arr_shape, dtype=type)
		return val_arr
	
	def make_null_feed_dict(self, data_dict):
		feed_dict = self._make_feed_dict(data_dict)
		for key, val in self._inputs.iteritems():
			if key not in data_dict:
				feed_dict[self._inputs[key]] = self._make_null_array(val.shape, val.dtype.as_numpy_dtype)
		return feed_dict
				
	def _make_feed_dict(self, data_dict):
		feed_dict = {}
		for key in data_dict.keys():
			
			if key in self._inputs:
				feed_dict[self._inputs[key]] = data_dict[key]
			else:pass
# 				print 'Unexpected argument {} in input dictionary!'.format(key)
		return feed_dict

	def _build_accuracy(self, decoded, targets, decoder_lengths):
		
		if isinstance(targets, tf.SparseTensor):
			targets = tf.sparse_tensor_to_dense(targets, default_value=-1)
			
# 		cut_len = tf.minimum(tf.shape(decoded)[1], tf.shape(targets)[1])
	
		
		with tf.variable_scope('accuracy_target'):
# 			flg = tf.equal(decoded[:, 0:cut_len],	targets[:, 0:cut_len])
			flg = tf.equal(decoded, targets[:, 0:tf.shape(decoded)[1]])
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
# 				 you can add your own options after them.
# 		note:do not change the default order and names of the first seven items.
# 		'''
# 		ops = [self.train_op, self.summary, self.global_step, self.output, self.loss, self.accuracy, self.seq_accuracy]
# 		names = [None, None, None, "global_step", "loss", "acc", "seq_acc"]
# 		
# 		return ops, names
	
	def run_ops(self, sess, data_dict, names=None, exclude_names=None):
		
		feed_dict = self.make_null_feed_dict(data_dict)
		
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
		
	def _add_eos(self, sequence, seq_lengths):
		
		batch_size = tf.shape(sequence)[0]
		pads = tf.ones([batch_size, 1], dtype=tf.int32) * self._PAD
		paded_sequence = tf.concat([sequence, pads], 1)
		max_decoder_time = tf.reduce_max(seq_lengths) + 1
		eos_sequence = paded_sequence[:, :max_decoder_time]

		eos = tf.one_hot(seq_lengths, depth=max_decoder_time,
								 on_value=self._EOS, off_value=self._PAD,
								 dtype=tf.int32)
		eos_sequence += eos
		return eos_sequence
	
	def _add_goes(self, sequence):
		batch_size = tf.shape(sequence)[0]
		goes = tf.ones([batch_size, 1], dtype=tf.int32) * self._START
		goes_sequence = tf.concat([goes, sequence], 1)
		return goes_sequence
	
	def _build_graph(self, num_features):
		self._build_inputs(num_features)
		targets = self.get_input('Y')
		

		logits, output, search_output = self._build_network_output()
		
		self._add_op(output, "output")
		
		self._add_op(search_output, "search_output")
		
		self._add_op(self._build_loss(logits, targets), "loss")
		
		self._add_op(self._build_train_step(self.get_op("loss")), "train_op")
		
		accuracy, seq_accuracy = self._build_accuracy(self.get_op("output"), self._add_eos(targets, self.get_input('Y_lenghts')), self.get_input('Y_lenghts') + 1)
		self._add_op(accuracy, "acc")
		self._add_op(seq_accuracy, "seq_acc")

		
		self._add_op(self._build_summary(), "summary")
		
		
		
		
		
		
		
