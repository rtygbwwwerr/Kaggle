import numpy as np # linear algebra
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.seq2seq import AttentionWrapper, AttentionWrapperState, \
									   BasicDecoder, BeamSearchDecoder, dynamic_decode, \
									   TrainingHelper, ScheduledEmbeddingTrainingHelper, sequence_loss, tile_batch, \
									   BahdanauAttention, LuongAttention
from tensorflow.python.layers import core as layers_core
from base.layers import BN_LSTMCell, Seq2SeqBase, ScheduledEmbeddingTrainingHelper_p
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import tensorflow.contrib.slim as slim
import math

from base.layers import ClassifierBase
from base.layers import depthwise_separable_conv2


epsilon = 1e-9

def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
	try:
		return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims, name=name)
	except:
		return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims, name=name)


def softmax(logits, axis=None, name=None):
	try:
		return tf.nn.softmax(logits, axis=axis, name=name)
	except:
		return tf.nn.softmax(logits, dim=axis, name=name)


def euclidean_norm(input, axis=2, keepdims=True, epsilon=True):
	if epsilon:
		norm = tf.sqrt(reduce_sum(tf.square(input), axis=axis, keepdims=keepdims) + 1e-9)
	else:
		norm = tf.sqrt(reduce_sum(tf.square(input), axis=axis, keepdims=keepdims))

	return(norm)

def get_transformation_matrix_shape(in_pose_shape, out_pose_shape):
	return([out_pose_shape[0], in_pose_shape[0]])

def spread_loss(labels, logits, margin, regularizer=None):
	'''
	Args:
		labels: [batch_size, num_label, 1].
		logits: [batch_size, num_label, 1].
		margin: Integer or 1-D Tensor.
		regularizer: use regularization.

	Returns:
		loss: Spread loss.
	'''
	# a_target: [batch_size, 1, 1]
	a_target = tf.matmul(labels, logits, transpose_a=True)
	dist = tf.maximum(0., margin - (a_target - logits))
	loss = tf.reduce_mean(tf.square(tf.matmul(1 - labels, dist, transpose_a=True)))
	if regularizer is not None:
		regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		loss += tf.reduce_mean(regularizer)
	return(loss)


def margin_loss():
	pass


def cross_entropy(labels, logits, regularizer=None):
	'''
	Args:
		...

	Returns:
		...
	'''
	loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
	if regularizer is not None:
		regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		loss += tf.reduce_mean(regularizer)
	return(loss)

def squash(vector):
	'''Squashing function
	Args:
		vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1]
	Returns:
		A tensor with the same shape as vector but squashed in 'vec_len' dimension.
	'''
	squared_norm = reduce_sum(tf.square(vector), axis=-2, keepdims=True)
	scalar_factor = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm + epsilon)
	return(scalar_factor * vector)


def routing(vote,
			activation=None,
			num_outputs=32,
			out_caps_shape=[4, 4],
			method='EMRouting',
			num_iter=3,
			regularizer=None):
	''' Routing-by-agreement algorithm.
	Args:
		alias H = out_caps_shape[0]*out_caps_shape[1].

		vote: [batch_size, num_inputs, num_outputs, H].
		activation: [batch_size, num_inputs, 1, 1].
		num_outputs: ...
		out_caps_shape: ...
		method: method for updating coupling coefficients between vote and pose['EMRouting', 'DynamicRouting'].
		num_iter: the number of routing iteration.
		regularizer: A (Tensor -> Tensor or None) function; the result of applying it on a newly created variable
				will be added to the collection tf.GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.

	Returns:
		pose: [batch_size, 1, 1, num_outputs] + out_caps_shape.
		activation: [batch_size, 1, 1, num_outputs].
	'''
	vote_stopped = tf.stop_gradient(vote, name="stop_gradient")
	batch_size = vote.shape[0].value
	if method == 'EMRouting':
		shape = vote.get_shape().as_list()[:3] + [1]
		# R: [batch_size, num_inputs, num_outputs, 1]
		R = tf.constant(np.ones(shape, dtype=np.float32) / num_outputs)
		for t_iter in range(num_iter):
			with tf.variable_scope('M-STEP') as scope:
				if t_iter > 0:
					scope.reuse_variables()
				# It's no need to do the `E-STEP` in the last iteration
				if t_iter == num_iter - 1:
					pose, stddev, activation_prime = M_step(R, activation, vote)
					break
				else:
					pose, stddev, activation_prime = M_step(R, activation, vote_stopped)
			with tf.variable_scope('E-STEP'):
				R = E_step(pose, stddev, activation_prime, vote_stopped)
		pose = tf.reshape(pose, shape=[batch_size, 1, 1, num_outputs] + out_caps_shape)
		activation = tf.reshape(activation_prime, shape=[batch_size, 1, 1, -1])
		return(pose, activation)
	elif method == 'DynamicRouting':
		B = tf.constant(np.zeros([batch_size, vote.shape[1].value, num_outputs, 1, 1], dtype=np.float32))
		for r_iter in range(num_iter):
			with tf.variable_scope('iter_' + str(r_iter)):
				coef = softmax(B, axis=2)
				if r_iter == num_iter - 1:
					s = reduce_sum(tf.multiply(coef, vote), axis=1, keepdims=True)
					pose = squash(s)
				else:
					s = reduce_sum(tf.multiply(coef, vote_stopped), axis=1, keepdims=True)
					pose = squash(s)
					shape = [batch_size, vote.shape[1].value, num_outputs] + out_caps_shape
					pose = tf.multiply(pose, tf.constant(1., shape=shape))
					B += tf.matmul(vote_stopped, pose, transpose_a=True)
		return(pose, activation)

	else:
		raise Exception('Invalid routing method!', method)


def M_step(R, activation, vote, lambda_val=0.9, regularizer=None):
	'''
	Args:
		alias H = out_caps_shape[0]*out_caps_shape[1]

		vote: [batch_size, num_inputs, num_outputs, H]
		activation: [batch_size, num_inputs, 1, 1]
		R: [batch_size, num_inputs, num_outputs, 1]
		lambda_val: ...

	Returns:
		pose & stddev: [batch_size, 1, num_outputs, H]
		activation: [batch_size, 1, num_outputs, 1]
	'''
	batch_size = vote.shape[0].value
	# line 2
	R = tf.multiply(R, activation)
	R_sum_i = tf.reduce_sum(R, axis=1, keepdims=True) + epsilon

	# line 3
	# mean: [batch_size, 1, num_outputs, H]
	pose = tf.reduce_sum(R * vote, axis=1, keepdims=True) / R_sum_i

	# line 4
	stddev = tf.sqrt(tf.reduce_sum(R * tf.square(vote - pose), axis=1, keepdims=True) / R_sum_i + epsilon)

	# line 5, cost: [batch_size, 1, num_outputs, H]
	H = vote.shape[-1].value
	beta_v = tf.get_variable('beta_v', shape=[batch_size, 1, pose.shape[2].value, H], regularizer=regularizer)
	cost = (beta_v + tf.log(stddev)) * R_sum_i

	# line 6
	beta_a = tf.get_variable('beta_a', shape=[batch_size, 1, pose.shape[2], 1], regularizer=regularizer)
	activation = tf.nn.sigmoid(lambda_val * (beta_a - tf.reduce_sum(cost, axis=3, keepdims=True)))

	return(pose, stddev, activation)


def E_step(pose, stddev, activation, vote):
	'''
	Args:
		alias H = out_caps_shape[0]*out_caps_shape[1]

		pose & stddev: [batch_size, 1, num_outputs, H]
		activation: [batch_size, 1, num_outputs, 1]
		vote: [batch_size, num_inputs, num_outputs, H]

	Returns:
		pose & var: [batch_size, 1, num_outputs, H]
		activation: [batch_size, 1, num_outputs, 1]
	'''
	# line 2
	var = tf.square(stddev)
	x = tf.reduce_sum(tf.square(vote - pose) / (2 * var), axis=-1, keepdims=True)
	peak_height = 1 / (tf.reduce_prod(tf.sqrt(2 * np.pi * var + epsilon), axis=-1, keepdims=True) + epsilon)
	P = peak_height * tf.exp(-x)

	# line 3
	R = tf.nn.softmax(activation * P, axis=2)
	return(R)

def fully_connected(inputs, activation,
					num_outputs,
					out_caps_shape,
					routing_method='EMRouting',
					reuse=None):
	'''A capsule fully connected layer.
	Args:
		inputs: A tensor with shape [batch_size, num_inputs] + in_caps_shape.
		activation: [batch_size, num_inputs]
		num_outputs: Integer, the number of output capsules in the layer.
		out_caps_shape: A list with two elements, pose shape of output capsules.
	Returns:
		pose: [batch_size, num_outputs] + out_caps_shape
		activation: [batch_size, num_outputs]
	'''
	in_pose_shape = inputs.get_shape().as_list()
	num_inputs = in_pose_shape[1]
	batch_size = in_pose_shape[0]
	T_size = get_transformation_matrix_shape(in_pose_shape[-2:], out_caps_shape)
	T_shape = [1, num_inputs, num_outputs] + T_size
	T_matrix = tf.get_variable("transformation_matrix", shape=T_shape)
	T_matrix = tf.tile(T_matrix, [batch_size, 1, 1, 1, 1])
	inputs = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, num_outputs, 1, 1])
	with tf.variable_scope('transformation'):
		# vote: [batch_size, num_inputs, num_outputs] + out_caps_shape
		vote = tf.matmul(T_matrix, inputs)
	with tf.variable_scope('routing'):
		if routing_method == 'EMRouting':
			activation = tf.reshape(activation, shape=activation.get_shape().as_list() + [1, 1])
			vote = tf.reshape(vote, shape=[batch_size, num_inputs, num_outputs, -1])
			pose, activation = routing(vote, activation, num_outputs, out_caps_shape, routing_method)
			pose = tf.reshape(pose, shape=[batch_size, num_outputs] + out_caps_shape)
			activation = tf.reshape(activation, shape=[batch_size, -1])
		elif routing_method == 'DynamicRouting':
			pose, _ = routing(vote, activation, num_outputs=num_outputs, out_caps_shape=out_caps_shape, method=routing_method)
			pose = tf.squeeze(pose, axis=1)
			activation = tf.squeeze(euclidean_norm(pose))
	return(pose, activation)


def primaryCaps(input, filters,
				kernel_size,
				strides,
				out_caps_shape,
				method=None,
				regularizer=None):
	'''PrimaryCaps layer
	Args:
		input: [batch_size, in_height, in_width, in_channels].
		filters: Integer, the dimensionality of the output space.
		kernel_size: ...
		strides: ...
		out_caps_shape: ...
		method: the method of calculating probability of entity existence(logistic, norm, None)
	Returns:
		pose: [batch_size, out_height, out_width, filters] + out_caps_shape
		activation: [batch_size, out_height, out_width, filters]
	'''
	# pose matrix
	pose_size = reduce(lambda x, y: x * y, out_caps_shape)
	pose = tf.layers.conv2d(input, filters * pose_size,
							kernel_size=kernel_size,
							strides=strides, activation=None,
							activity_regularizer=regularizer)
	pose_shape = pose.get_shape().as_list()[:3] + [filters] + out_caps_shape
	pose = tf.reshape(pose, shape=pose_shape)

	if method == 'logistic':
		# logistic activation unit
		activation = tf.layers.conv2d(input, filters,
									  kernel_size=kernel_size,
									  strides=strides,
									  activation=tf.nn.sigmoid,
									  activity_regularizer=regularizer)
	elif method == 'norm':
		activation = euclidean_norm(pose)
	else:
		activation = None

	return(pose, activation)


def conv2d(in_pose,
		   activation,
		   filters,
		   out_caps_shape,
		   kernel_size,
		   strides=(1, 1),
		   coordinate_addition=False,
		   regularizer=None,
		   reuse=None):
	'''A capsule convolutional layer.
	Args:
		in_pose: A tensor with shape [batch_size, in_height, in_width, in_channels] + in_caps_shape.
		activation: A tensor with shape [batch_size, in_height, in_width, in_channels]
		filters: ...
		out_caps_shape: ...
		kernel_size: ...
		strides: ...
		coordinate_addition: ...
		regularizer: apply regularization on a newly created variable and add the variable to the collection tf.GraphKeys.REGULARIZATION_LOSSES.
		reuse: ...
	Returns:
		out_pose: A tensor with shape [batch_size, out_height, out_height, out_channals] + out_caps_shape,
		out_activation: A tensor with shape [batch_size, out_height, out_height, out_channels]
	'''
	# do some preparation stuff
	in_pose_shape = in_pose.get_shape().as_list()
	in_caps_shape = in_pose_shape[-2:]
	batch_size = in_pose_shape[0]
	in_channels = in_pose_shape[3]

	T_size = get_transformation_matrix_shape(in_caps_shape, out_caps_shape)
	if isinstance(kernel_size, int):
		h_kernel_size = kernel_size
		w_kernel_size = kernel_size
	elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2:
		h_kernel_size = kernel_size[0]
		w_kernel_size = kernel_size[1]
	if isinstance(strides, int):
		h_stride = strides
		w_stride = strides
	elif isinstance(strides, (list, tuple)) and len(strides) == 2:
		h_stride = strides[0]
		w_stride = strides[1]
	num_inputs = h_kernel_size * w_kernel_size * in_channels
	batch_shape = [batch_size, h_kernel_size, w_kernel_size, in_channels]
	T_shape = (1, num_inputs, filters) + tuple(T_size)

	T_matrix = tf.get_variable("transformation_matrix", shape=T_shape, regularizer=regularizer)
	T_matrix_batched = tf.tile(T_matrix, [batch_size, 1, 1, 1, 1])

	h_step = int((in_pose_shape[1] - h_kernel_size) / h_stride + 1)
	w_step = int((in_pose_shape[2] - w_kernel_size) / w_stride + 1)
	out_pose = []
	out_activation = []
	# start to do capsule convolution.
	# Note: there should be another way more computationally efficient to do this
	for i in range(h_step):
		col_pose = []
		col_prob = []
		h_s = i * h_stride
		h_e = h_s + h_kernel_size
		for j in range(w_step):
			with tf.variable_scope("transformation"):
				begin = [0, i * h_stride, j * w_stride, 0, 0, 0]
				size = batch_shape + in_caps_shape
				w_s = j * w_stride
				pose_sliced = in_pose[:, h_s:h_e, w_s:(w_s + w_kernel_size), :, :, :]
				pose_reshaped = tf.reshape(pose_sliced, shape=[batch_size, num_inputs, 1] + in_caps_shape)
				shape = [batch_size, num_inputs, filters] + in_caps_shape
				batch_pose = tf.multiply(pose_reshaped, tf.constant(1., shape=shape))
				vote = tf.reshape(tf.matmul(T_matrix_batched, batch_pose), shape=[batch_size, num_inputs, filters, -1])
				# do Coordinate Addition. Note: not yet completed
				if coordinate_addition:
					x = j / w_step
					y = i / h_step

			with tf.variable_scope("routing") as scope:
				if i > 0 or j > 0:
					scope.reuse_variables()
				begin = [0, i * h_stride, j * w_stride, 0]
				size = [batch_size, h_kernel_size, w_kernel_size, in_channels]
				prob = tf.slice(activation, begin, size)
				prob = tf.reshape(prob, shape=[batch_size, -1, 1, 1])
				pose, prob = routing(vote, prob, filters, out_caps_shape, method="EMRouting", regularizer=regularizer)
			col_pose.append(pose)
			col_prob.append(prob)
		col_pose = tf.concat(col_pose, axis=2)
		col_prob = tf.concat(col_prob, axis=2)
		out_pose.append(col_pose)
		out_activation.append(col_prob)
	out_pose = tf.concat(out_pose, axis=1)
	out_activation = tf.concat(out_activation, axis=1)

	return(out_pose, out_activation)


# # TODO: 1. Test the `fully_connected` and `conv2d` function;
# #	   2. Update  docs about these two function.
# def fully_connected(inputs,
# 					num_outputs,
# 					vec_len,
# 					with_routing=True,
# 					weights_initializers=tf.contrib.layers.xavier_initializer(),
# 					reuse=None,
# 					variable_collections=None,
# 					scope=None):
# 	'''A capsule fully connected layer.(Note: not tested yet)
# 	Args:
# 		inputs: A tensor of as least rank 3, i.e. `[batch_size, num_inputs, vec_len]`,
# 				`[batch_size, num_inputs, vec_len, 1]`.
# 		num_outputs: ...
# 	Returns:
# 		...
# 	Raise:
# 		...
# 	'''
# 	layer = CapsLayer(num_outputs=num_outputs,
# 					  vec_len=vec_len,
# 					  with_routing=with_routing,
# 					  layer_type='FC')
# 	return layer.apply(inputs)


# def conv2d(inputs,
# 		   filters,
# 		   vec_len,
# 		   kernel_size,
# 		   strides=(1, 1),
# 		   with_routing=False,
# 		   reuse=None):
# 	'''A capsule convolutional layer.(Note: not tested yet)
# 	Args:
# 		inputs: A tensor.
# 	Returns:
# 		...
# 	Raises:
# 		...
# 	'''
# 	layer = CapsLayer(num_outputs=filters,
# 					  vec_len=vec_len,
# 					  with_routing=with_routing,
# 					  layer_type='CONV')
# 	return(layer(inputs, kernel_size=kernel_size, stride=strides))

class CapsNet(object):
	def __init__(self, num_label, width, height, channels=1, batch_size=256, is_training=True, 
				m_plus=0.9, m_minus=0.1, lambda_val=0.5, iter_routing=3, stddev=0.1, 
				regularization_rate=0.000125, routing_method='DynamicRouting',
				):
		'''
		Args:
			height: ...
			width: ...
			channels: ...
		'''
		self.batch_size = batch_size
		self.height = height
		self.width = width
		self.channels = channels
		self.num_label = num_label
		
		self.m_plus = m_plus
		self.m_minus = m_minus
		self.lambda_val = lambda_val
		self.iter_routing = iter_routing
		self.stddev = stddev
		self.regularization_scale = regularization_rate * width * height * 0.004
		print "regularization_scale:{}".format(self.regularization_scale)

		self._inputs = {}
		self._build_inputs()
		self._build_graph(is_training, routing_method)



	def _build_accuracy(self):
		with tf.variable_scope('accuracy'):
			labels = tf.argmax(self._inputs['Y'], axis=1)
			logits_idx = tf.to_int32(tf.argmax(softmax(self.activation, axis=1), axis=1))
			correct_prediction = tf.equal(tf.to_int32(labels), logits_idx)
			self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) / float(self.batch_size)
			self.test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])
	
	def _build_graph(self, is_training, routing_method):
		
# 		self.graph = tf.Graph()
# 		with self.graph.as_default():
		if is_training:

			self.build_arch(routing_method)
			self._build_accuracy()
			self.loss()
			self._summary()

			self.global_step = tf.Variable(1, name='global_step', trainable=False)
# 			self.optimizer = tf.train.AdamOptimizer()
# 			self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1)


			lr = tf.train.exponential_decay(
				self._inputs["init_lr_rate"],
				self.global_step,
				self._inputs["decay_step"],
				self._inputs["decay_factor"],
				staircase=True
			)
			lr = tf.clip_by_value(
				lr,
				1e-4,
				self._inputs["init_lr_rate"],
				name='lr_clip'
			)
			self.lr = lr

			self.optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
			self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
		else:
			self.build_arch(routing_method)


			
	def _build_inputs(self):
		
# 		self._batch_size = tf.placeholder(shape=(), dtype=tf.int32, name='batch_size')
# 		batch_size = self._batch_size

		self._inputs['X'] = tf.placeholder(
			shape=(None, None, None), # batch_size, max_time, feature_dim
			dtype=tf.float32,
			name='X'
		)
		
		
		
		self._inputs['Y'] = tf.placeholder(
			shape=(None, None), # batch_size, max_time
			dtype=tf.int32,
			name='Y'
		)
		
		self._inputs['init_lr_rate'] = tf.placeholder(
			dtype=tf.float32,
			name='init_lr_rate'
		)
		
		self._inputs['decay_step'] = tf.placeholder(
			dtype=tf.int32,
			name='decay_step'
		)
		
		self._inputs['decay_factor'] = tf.placeholder(
			dtype=tf.float32,
			name='decay_factor'
		)


# 		else:
# 			inputs, constract_inputs = input_batch
# 			inputs.set_shape([None, None])
# 			constract_inputs.set_shape([None, None])
# 
# 			self._inputs = {
# 				'inputs': inputs,
# 				'constract_inputs': constract_inputs,
# 			}

# 		return self._inputs['X'], self._inputs['Y'], \
# 				self._inputs['init_lr_rate'], \
# 			   self._inputs['decay_step'], self._inputs['decay_factor']
			
	def make_feed_dict(self, data_dict):
		feed_dict = {}
		for key in data_dict.keys():
			try:
				feed_dict[self._inputs[key]] = data_dict[key]
			except KeyError:
				raise ValueError('Unexpected argument in input dictionary!')
		return feed_dict
	
	def build_arch(self, routing_method):
		
		x = tf.reshape(self._inputs['X'], shape=[self.batch_size, self.height, self.width, self.channels])
		Y = self._inputs['Y']
		with tf.variable_scope('Conv1_layer'):
			# Conv1, return with shape [batch_size, 20, 20, 256]
			conv1 = tf.contrib.layers.conv2d(x, num_outputs=256, kernel_size=2, stride=2, padding='VALID')

		# return primaryCaps: [batch_size, 1152, 8, 1], activation: [batch_size, 1152]
		with tf.variable_scope('PrimaryCaps_layer'):
			primary, activation = primaryCaps(conv1, filters=32, kernel_size=5, strides=2, out_caps_shape=[6, 1])

		# return digitCaps: [batch_size, 10, 16, 1], activation: [batch_size, 10]
		with tf.variable_scope('DigitCaps_layer'):
			primary = tf.reshape(primary, shape=[self.batch_size, -1, 6, 1])
			self.digitCaps, self.activation = fully_connected(primary, activation, num_outputs=self.num_label, out_caps_shape=[8, 1], routing_method=routing_method)

		# Decoder structure in Fig. 2
		# Reconstructe the MNIST images with 3 FC layers
		# [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
		Y = tf.cast(Y, dtype=tf.float32, name=None)
		y = tf.reshape(Y, (-1, self.num_label, 1, 1))
		with tf.variable_scope('Decoder'):
			masked_caps = tf.multiply(self.digitCaps, y)
			active_caps = tf.reshape(masked_caps, shape=(self.batch_size, -1))
			fc1 = tf.contrib.layers.fully_connected(active_caps, num_outputs=512)
			fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
			self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=self.width*self.height, activation_fn=tf.sigmoid)

	def loss(self):
		x = tf.reshape(self._inputs['X'], shape=[self.batch_size, self.height, self.width, self.channels])
		Y = self._inputs['Y']
		
		# 1. Margin loss

		# [batch_size, 10, 1, 1]
		# max_l = max(0, m_plus-||v_c||)^2
		max_l = tf.square(tf.maximum(0., self.m_plus - self.activation))
		# max_r = max(0, ||v_c||-m_minus)^2
		max_r = tf.square(tf.maximum(0., self.activation - self.m_minus))
		assert max_l.get_shape() == [self.batch_size, self.num_label]

		# reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
		max_l = tf.reshape(max_l, shape=(self.batch_size, -1))
		max_r = tf.reshape(max_r, shape=(self.batch_size, -1))

		# calc T_c: [batch_size, 10]
		# T_c = Y, is my understanding correct? Try it.
		T_c = tf.cast(Y, dtype=tf.float32, name=None)
		# [batch_size, 10], element-wise multiply
		L_c = T_c * max_l + self.lambda_val * (1 - T_c) * max_r

		self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

		# 2. The reconstruction loss
		orgin = tf.reshape(x, shape=(self.batch_size, -1))
		squared = tf.square(self.decoded - orgin)
		self.reconstruction_err = tf.reduce_mean(squared)

		# 3. Total loss
		# The paper uses sum of squared error as reconstruction error, but we
		# have used reduce_mean in `# 2 The reconstruction loss` to calculate
		# mean squared error. In order to keep in line with the paper,the
		# regularization scale should be 0.0005*784=0.392
		self.loss = self.margin_loss + self.regularization_scale * self.reconstruction_err
		
# 		self.loss = cross_entropy(tf.argmax(self._inputs['Y'], axis=1), tf.to_int32(tf.argmax(softmax(self.activation, axis=1), axis=1)))

	# Summary
	def _summary(self):
		train_summary = []
		train_summary.append(tf.summary.scalar('acc', self.accuracy))
		train_summary.append(tf.summary.scalar('margin_loss', self.margin_loss))
		train_summary.append(tf.summary.scalar('reconstruction_loss', self.reconstruction_err))
		train_summary.append(tf.summary.scalar('total_loss', self.loss))
# 		recon_img = tf.reshape(self.decoded, shape=(self.batch_size, 28, 28, 1))
# 		train_summary.append(tf.summary.image('reconstruction_img', recon_img))
# 		train_summary.append(tf.summary.audio("recon_audio", tensor, sample_rate, max_outputs, collections, family))
		train_summary.append(tf.summary.histogram('activation', self.activation))
		self.train_summary = tf.summary.merge(train_summary)
		
	def restore_from_session(self, sess):
		print "please implement!"
		
		
def make_CapsNet(num_label, width, height, batch_size=256, is_training=True):
	return CapsNet(num_label, width, height, batch_size=batch_size, is_training=is_training)



	
class Seq2SeqCTCModel(Seq2SeqBase):
	

	def __init__(self, 
				encoder_hidden_size, 
				decoder_hidden_size, 
				embedding_dim, 
				vocab_size, 
				input_feature_num,
				is_training = True,
				is_restored = False,
				encoder_cell_type = 'BN_LSTM',
				decoder_cell_type = 'LSTM',
				n_encoder_layers = 2,
				n_decoder_layers = 1,
				attention_type = 'Bahdanau',
				attention_num_units=100,
				max_decode_iter_size=10,
				init_learning_rate=0.01,
				minimum_learning_rate=1e-4,
# 				decay_steps=2e4,
# 				decay_factor=0.3,
				attention_depth=100,
				is_bidrection=True,
				is_attention=True,
				beam_width=5,
				PAD=0,
				START=1,
				EOS=2):
		
		super(Seq2SeqCTCModel, self).__init__()
		
		self._is_training = is_training
		self._PAD = PAD
 		self._START = START
		self._EOS = EOS
		self._batch_size = None
		self._embedding_dim = embedding_dim
		self._vocab_size = vocab_size
# 		self._dropout = dropout
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
# 		self._decay_steps = decay_steps
# 		self._decay_factor = decay_factor
		self._minimum_learning_rate = minimum_learning_rate
		
		
		if not is_restored:
			self._build_graph(input_feature_num)
			
			
	def _build_inputs(self, num_features):
		
		super(Seq2SeqCTCModel, self)._build_inputs(num_features, sparse_y=True)
		
		
		self._inputs['decay_step'] = tf.placeholder(
			dtype=tf.int32,
			name='decay_step'
		)
		
		self._inputs['decay_factor'] = tf.placeholder(
			dtype=tf.float32,
			name='decay_factor'
		)

		
	def _build_network_output(self):
		inputs = self._inputs['X']
		seq_len = self._inputs['X_lenghts']
		
		num_hidden = self._encoder_hidden_size
		num_classes = self._vocab_size
		# Defining the cell
		# Can be:
		#   tf.nn.rnn_cell.RNNCell
		#   tf.nn.rnn_cell.GRUCell
		cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
		
		# Stacking rnn cells
		stack = tf.contrib.rnn.MultiRNNCell([cell] * self._n_encoder_layers, state_is_tuple=True)
		
		# The second output is the last state and we will no use that
		outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
		
		shape = tf.shape(inputs)
		batch_s, max_timesteps = shape[0], shape[1]
		
		# Reshaping to apply the same weights over the timesteps
		outputs = tf.reshape(outputs, [-1, num_hidden])
		
		# Truncated normal with mean 0 and stdev=0.1
		# Tip: Try another initialization
		# see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
		W = tf.Variable(tf.truncated_normal([num_hidden,
		                                     num_classes],
		                                    stddev=0.1))
		# Zero initialization
		# Tip: Is tf.zeros_initializer the same?
		b = tf.Variable(tf.constant(0., shape=[num_classes]))
		
		# Doing the affine projection
		logits = tf.matmul(outputs, W) + b
		
		# Reshaping back to the original shape
		logits = tf.reshape(logits, [batch_s, -1, num_classes])
		
		# Time major, because the ctc decoder only support time major mode
		logits = tf.transpose(logits, (1, 0, 2))
		
		# Option 2: tf.nn.ctc_beam_search_decoder
		# (it's slower but you'll get better results)
		decoded_sparse_tuple, log_prob = tf.nn.ctc_greedy_decoder(logits, self._inputs['X_lenghts'])
		dense_decoded = tf.sparse_tensor_to_dense(decoded_sparse_tuple[0], default_value=-1)
		output = tf.cast(dense_decoded, dtype=tf.int32)
	
		return logits, output, output
		
	def _build_loss(self, logits, targets):
		loss = tf.nn.ctc_loss(targets, logits, self._inputs['X_lenghts'])
		loss = tf.reduce_mean(loss)
		return loss
	
	def _build_train_step(self, cost):
		
		opt = tf.train.MomentumOptimizer(self._inputs['init_lr_rate'], 0.9).minimize(cost, global_step=self.get_op("global_step"))
		with tf.control_dependencies([opt]):
			train_op = tf.no_op(name='train_op')
		return train_op

	def _build_summary(self):
		
		train_summary = []
		train_summary.append(tf.summary.scalar('acc', self.get_op("acc")))
		train_summary.append(tf.summary.scalar('seq_acc', self.get_op("seq_acc")))
		train_summary.append(tf.summary.scalar('loss', self.get_op("loss")))
		return tf.summary.merge(train_summary)
		
		
	
def make_tf_CTCSeq2seq(
				encoder_hidden_size, 
				decoder_hidden_size, 
				batch_size, 
				embedding_dim, 
				vocab_size, 
				input_feature_num,
				max_decode_iter_size,
				is_training = True,
				is_restored = False,
				dropout = 0.9,
				encoder_cell_type = 'BN_LSTM',
				decoder_cell_type = 'LSTM',
				n_encoder_layers = 1,
				n_decoder_layers = 1,
				attention_type = 'Bahdanau',
				attention_num_units=100,
# 				init_learning_rate=0.35,
				minimum_learning_rate=1e-5,
# 				decay_steps=3e4,
# 				decay_factor=0.6,
				attention_depth=100,
				is_bidrection=True,
				is_attention=True,
				beam_width=5,
				PAD=0, 
				START=1, 
				EOS=2,
				
		):
	return Seq2SeqCTCModel(
				encoder_hidden_size = encoder_hidden_size, 
				decoder_hidden_size = decoder_hidden_size, 
				batch_size = batch_size, 
				embedding_dim = embedding_dim, 
				vocab_size = vocab_size, 
				input_feature_num = input_feature_num,
				is_restored = is_restored,
				is_training = is_training,
				dropout = dropout,
				encoder_cell_type = encoder_cell_type,
				decoder_cell_type = decoder_cell_type,
				n_encoder_layers = n_encoder_layers,
				n_decoder_layers = n_decoder_layers,
				attention_type = attention_type,
				attention_num_units = attention_num_units,
				max_decode_iter_size = max_decode_iter_size,
# 				init_learning_rate = init_learning_rate,
				minimum_learning_rate = minimum_learning_rate,
# 				decay_steps = decay_steps,
# 				decay_factor = decay_factor,
				attention_depth = attention_depth,
				is_bidrection = is_bidrection,
				is_attention = is_attention,
				beam_width = beam_width,
				PAD = PAD, 
				EOS = EOS, 
				START = START,
				input_batch=None)	


def make_tf_AttentionSeq2Seq(
				encoder_hidden_size, 
				decoder_hidden_size, 
				embedding_dim, 
				vocab_size, 
				input_feature_num,
				max_decode_iter_size,
				is_training = True,
				is_restored = False,
				encoder_cell_type = 'BN_LSTM',
				decoder_cell_type = 'LSTM',
				n_encoder_layers = 1,
				n_decoder_layers = 1,
				attention_type = 'Bahdanau',
				attention_num_units=100,
# 				init_learning_rate=0.35,
				minimum_learning_rate=1e-5,
# 				decay_steps=3e4,
# 				decay_factor=0.6,
				attention_depth=100,
				is_bidrection=True,
				is_attention=True,
				beam_width=5,
				PAD=0, 
				START=1, 
				EOS=2,
				
		):
	return Seq2SeqAttentionModel(
				encoder_hidden_size = encoder_hidden_size, 
				decoder_hidden_size = decoder_hidden_size, 
				embedding_dim = embedding_dim, 
				vocab_size = vocab_size, 
				input_feature_num = input_feature_num,
				is_restored = is_restored,
				is_training = is_training,
				encoder_cell_type = encoder_cell_type,
				decoder_cell_type = decoder_cell_type,
				n_encoder_layers = n_encoder_layers,
				n_decoder_layers = n_decoder_layers,
				attention_type = attention_type,
				attention_num_units = attention_num_units,
				max_decode_iter_size = max_decode_iter_size,
# 				init_learning_rate = init_learning_rate,
				minimum_learning_rate = minimum_learning_rate,
# 				decay_steps = decay_steps,
# 				decay_factor = decay_factor,
				attention_depth = attention_depth,
				is_bidrection = is_bidrection,
				is_attention = is_attention,
				beam_width = beam_width,
				PAD = PAD, 
				EOS = EOS, 
				START = START)
	
		
class Seq2SeqAttentionModel(Seq2SeqBase):
	
	def __init__(self, 
				encoder_hidden_size, 
				decoder_hidden_size, 
				embedding_dim, 
				vocab_size, 
				input_feature_num,
				is_training = True,
				is_restored = False,
				encoder_cell_type = 'BN_LSTM',
				decoder_cell_type = 'LSTM',
				n_encoder_layers = 2,
				n_decoder_layers = 1,
				attention_type = 'Bahdanau',
				attention_num_units=100,
				max_decode_iter_size=10,
				init_learning_rate=0.01,
				minimum_learning_rate=1e-4,
# 				decay_steps=2e4,
# 				decay_factor=0.3,
				attention_depth=100,
				is_bidrection=True,
				is_attention=True,
				beam_width=10,
				PAD=0,
				START=1,
				EOS=2):
		
		super(Seq2SeqAttentionModel, self).__init__()
		
		self._is_training = is_training
		self._PAD = PAD
 		self._START = START
		self._EOS = EOS
		self._batch_size = None
		self._embedding_dim = embedding_dim
		self._vocab_size = vocab_size
# 		self._dropout = dropout
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
# 		self._decay_steps = decay_steps
# 		self._decay_factor = decay_factor
		self._minimum_learning_rate = minimum_learning_rate

		
		self._build_graph(input_feature_num)
			
	def _build_inputs(self, num_features):
		
		super(Seq2SeqAttentionModel, self)._build_inputs(num_features)
		
		self.set_input('decay_step', tf.placeholder(dtype=tf.int32, name='decay_step'))
		
		self.set_input('decay_factor', tf.placeholder(dtype=tf.float32, name='decay_factor'))
		self.set_input('keep_output_rate', tf.placeholder(dtype=tf.float32, name='keep_output_rate'))
		self.set_input('sampling_probability', tf.placeholder(dtype=tf.float32, name='sampling_probability'))
		

		
	def _build_loss(self, logits, targets):
		with tf.variable_scope('loss_target'):
			#build decoder output, with appropriate padding and mask
# 			batch_size = tf.shape(targets)[0]
# 			pads = tf.ones([batch_size, 1], dtype=tf.int32) * self._PAD
# 			paded_decoder_inputs = tf.concat([targets, pads], 1)
# 			max_decoder_time = tf.reduce_max(self.get_input('Y_lenghts')) + 1
# # 			max_decoder_time = tf.reduce_max(self.get_input('Y_lenghts'))
# 			decoder_target = paded_decoder_inputs[:, :max_decoder_time]
# 
# 			decoder_eos = tf.one_hot(self.get_input('Y_lenghts'), depth=max_decoder_time,
# 									 on_value=self._EOS, off_value=self._PAD,
# 									 dtype=tf.int32)
			max_decoder_time = tf.reduce_max(self.get_input('Y_lenghts')) + 1
			decoder_target = self._add_eos(targets, self.get_input('Y_lenghts'))

			decoder_loss_mask = tf.sequence_mask(self.get_input('Y_lenghts') + 1,
												 maxlen=max_decoder_time,
												 dtype=tf.float32)

		with tf.variable_scope('loss'):
			seq_loss = sequence_loss(
				logits,
				decoder_target,
				decoder_loss_mask,
				name='sequence_loss'
			)
			
# 			seq_loss = tf.nn.ctc_loss(labels = decoder_target, 
# 									inputs = decoder_logits, 
# 									decoder_lengths, 
# 									preprocess_collapse_repeated, 
# 									ctc_merge_repeated,
# 									ignore_longer_outputs_than_inputs, 
# 									time_major)

		return seq_loss
			
	def _build_network_output(self):
		encoder_inputs = self.get_input('X')
		encoder_lengths = self.get_input('X_lenghts')
		
		decoder_inputs = self.get_input('Y')
		decoder_lengths = self.get_input('Y_lenghts')
		self._build_sampling_schedule()
		encoder_outputs, encoder_final_state = self._build_encoder(encoder_inputs, encoder_lengths)
		decoder_results = self._build_decoder(encoder_outputs, encoder_final_state, encoder_lengths, decoder_inputs, decoder_lengths)
		
		return decoder_results['decoder_outputs'], decoder_results['decoder_result_ids'], decoder_results['beam_decoder_result_ids'][:, :, 0]
	
	
	def _build_sampling_schedule(self, decay_steps=10000, learning_rate=0.9, offest=5.0, staircase=True):
		global_step = self.get_op("global_step")
		init_sp = self.get_input('sampling_probability')
		with tf.variable_scope('sampling_schedule'):
			sp_prob = tf.cast(global_step, tf.float32) * init_sp
			sp_prob = tf.minimum(1.0, sp_prob)
			sp_prob.set_shape(())
# 		with tf.variable_scope('sampling_schedule'):
# 			learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
# 			dtype = learning_rate.dtype
# 			global_step = math_ops.cast(global_step, dtype)
# 			decay_steps = math_ops.cast(decay_steps, dtype)
# 
# 			p = global_step / decay_steps
# 			if staircase:
# 				p = math_ops.floor(p)
# 			with tf.variable_scope('sigmoid_sampling'):
# 				sigmoid_op = 1.0 / (1.0 + math_ops.exp(offest + (-learning_rate * p)))
# 				sigmoid_op = tf.add(sigmoid_op, self._init_sampling_prob)
# 				sigmoid_op = tf.cond(sigmoid_op > tf.constant(0.00015,dtypes.float32), lambda: sigmoid_op, lambda: tf.constant(0.0, dtypes.float32))
# 				sigmoid_op = tf.cond(sigmoid_op <= tf.constant(0.99,dtypes.float32), lambda: sigmoid_op, lambda: tf.constant(1.0, dtypes.float32))
# 				sp_prob = math_ops.cast(sigmoid_op, dtype)
# 				sp_prob.set_shape(())
# 				print sp_prob.get_shape()
			self._add_op(sp_prob, "sp")
	
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
				self._minimum_learning_rate,
				self.get_input('init_lr_rate'),
				name='lr_clip'
			)

# 			opt = tf.train.GradientDescentOptimizer(self._lr)
# 			opt = tf.train.MomentumOptimizer(self.lr, 0.9)
# 			opt = tf.train.RMSPropOptimizer(learning_rate=0.05)
			opt = tf.train.AdadeltaOptimizer(learning_rate=lr)
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
		tf.summary.scalar('sampling_rate', self.get_op('sp'))
		tf.summary.scalar('learning_rate', self.get_op('lr'))
		tf.summary.scalar('loss', self.get_op('loss'))
		tf.summary.scalar('accuracy', self.get_op('acc'))
		tf.summary.scalar('accuracy_seqs', self.get_op('seq_acc'))
		return tf.summary.merge_all()
	
	
	def _build_encoder(self, encoder_inputs, encoder_lengths):


			# batch_size, max_time, embed_dims
		encoder_input_vectors = encoder_inputs

		with tf.variable_scope('encoder'):
			if self._is_bidrection:
				fw_cells, bw_cells = self._create_biRNNLayers(self._encoder_hidden_size, self.get_input('keep_output_rate'), 
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
				cells = self._create_RNNLayers(self._encoder_hidden_size, self.get_input('keep_output_rate'), 
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
			
			cells = tf.nn.rnn_cell.DropoutWrapper(cell=cells, input_keep_prob=1.0, output_keep_prob=self.get_input('keep_output_rate'))
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

# 		with tf.variable_scope('word_embedding', reuse=True):
# 			word_embedding = tf.get_variable(name="word_embedding")
			
		with tf.variable_scope('word_embedding'):
			word_embedding = tf.get_variable(
				name="word_embedding",
				shape=(self._vocab_size, self._embedding_dim),
				initializer=xavier_initializer(),
				dtype=tf.float32
			)

		with tf.variable_scope('decoder'):
			out_func = layers_core.Dense(
				self._vocab_size, use_bias=False)

			goes = tf.ones([batch_size, 1], dtype=tf.int32) * self._START
# 			goes_decoder_inputs = decoder_inputs
			goes_decoder_inputs = tf.concat([goes, decoder_inputs], 1)
			
			#using teaching force without schedule			
			embed_decoder_inputs = tf.nn.embedding_lookup(word_embedding, goes_decoder_inputs)
# 
# 			training_helper = TrainingHelper(
# 				embed_decoder_inputs,
# 				decoder_lengths + 1
# 			)

			
			
# 			def embedding_lookup(ids):
# 				tf.nn.embedding_lookup(word_embedding, goes_decoder_inputs)

			training_helper = ScheduledEmbeddingTrainingHelper_p(
				embed_decoder_inputs,
				decoder_lengths + 1,
				word_embedding,
				self.get_op('sp'),
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
			#notice: if we use decoder with schedule sampling, many invalid flg "-1" would be set
			# into the output_ids vectors, thus, we have to restore the real output_ids through argmax operation 
# 			'decoder_result_ids': decoder_outputs[1],
			'decoder_result_ids': math_ops.cast(math_ops.argmax(decoder_outputs[0], axis=-1), dtypes.int32),
			'decoder_state': decoder_state,
			'decoder_sequence_lengths': decoder_sequence_lengths,
			'beam_decoder_result_ids': beam_decoder_outputs.predicted_ids,
			'beam_decoder_scores': beam_decoder_outputs.beam_search_decoder_output.scores,
			'beam_decoder_state': beam_decoder_state,
			'beam_decoder_sequence_outputs': beam_decoder_sequence_lengths
		}
		
		
		return decoder_results
				
	def _create_RNNLayers(self, state_size, keep_output_rate, n_encoder_layers, is_training):
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
		
		cells = tf.nn.rnn_cell.DropoutWrapper(cell=cells, input_keep_prob=1.0, output_keep_prob=keep_output_rate)
		cells  = tf.nn.rnn_cell.MultiRNNCell([cells] * n_encoder_layers)
		return cells
	
	def _create_biRNNLayers(self, state_size, keep_output_rate, n_encoder_layers, is_training):
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
		
		fw_cells = tf.nn.rnn_cell.DropoutWrapper(cell=fw_cells, input_keep_prob=1.0, output_keep_prob=keep_output_rate)
		bw_cells = tf.nn.rnn_cell.DropoutWrapper(cell=bw_cells, input_keep_prob=1.0, output_keep_prob=keep_output_rate)
		
		fw_cells  = tf.nn.rnn_cell.MultiRNNCell([fw_cells] * n_encoder_layers)
		bw_cells  = tf.nn.rnn_cell.MultiRNNCell([bw_cells] * n_encoder_layers)
		return fw_cells, bw_cells


def make_tf_dscnn(input_info, model_info):
	return DSCNN(input_info, model_info)

class DSCNN(ClassifierBase):
	"""Builds a model with depthwise separable convolutional neural network
	Model definition is based on https://arxiv.org/abs/1704.04861 and
	Tensorflow implementation: https://github.com/Zehaos/MobileNet

	model_size_info: defines number of layers, followed by the DS-Conv layer
		parameters in the order {number of conv features, conv filter height, 
		width and stride in y,x dir.} for each of the layers. 
	Note that first layer is always regular convolution, but the remaining 
		layers are all depthwise separable convolutions.
		"""
		
	def __init__(self, input_info, model_info):
		super(DSCNN, self).__init__(input_info, model_info)
		self._is_training = input_info['is_training']
		self._build_graph()
		
	def _gen_cnn_layer_info(self, model_size_info):
		num_layers = model_size_info[0]
		conv_feat = [None]*num_layers
		conv_kt = [None]*num_layers
		conv_kf = [None]*num_layers
		conv_st = [None]*num_layers
		conv_sf = [None]*num_layers
		i=1
		
		for layer_no in range(0, num_layers):
			conv_feat[layer_no] = model_size_info[i]
			i += 1
			conv_kt[layer_no] = model_size_info[i]
			i += 1
			conv_kf[layer_no] = model_size_info[i]
			i += 1
			conv_st[layer_no] = model_size_info[i]
			i += 1
			conv_sf[layer_no] = model_size_info[i]
			i += 1
		return num_layers, conv_feat, conv_kt, conv_kf, conv_st, conv_sf
	
	def _gen_dscnn_structure(self, sub_id):
		
		model_size_info = self._model_info["model_size_infos"][sub_id]
		num_layers, conv_feat, conv_kt, conv_kf, conv_st, conv_sf = self._gen_cnn_layer_info(model_size_info)
		x_dim = self._input_info['x_dims'][sub_id]
		t_dim = x_dim[0]
		f_dim = x_dim[1]
		input_shape = tf.reshape(self.get_input(ClassifierBase.get_x_name(sub_id)),
									[-1, x_dim[0], x_dim[1], 1])
		with tf.variable_scope('DS-CNN{}'.format(sub_id)) as sc:
			end_points_collection = sc.name + '_end_points'
			with slim.arg_scope([slim.convolution2d,
								 slim.separable_convolution2d],
								activation_fn=None,
								weights_initializer=slim.initializers.xavier_initializer(),
								biases_initializer=slim.init_ops.zeros_initializer(),
								outputs_collections=[end_points_collection]):
				with slim.arg_scope([slim.batch_norm],
									is_training=self._is_training,
									decay=0.96,
									updates_collections=None,
									activation_fn=tf.nn.relu):
					for layer_no in range(num_layers):
						if layer_no==0:
							net = slim.convolution2d(input_shape, conv_feat[layer_no],\
												[conv_kt[layer_no], conv_kf[layer_no]], stride=[conv_st[layer_no], conv_sf[layer_no]], padding='SAME', scope='conv_1')
							net = slim.batch_norm(net, scope='conv/batch_norm')
						else:
							net = depthwise_separable_conv2(net, conv_feat[layer_no], 
												kernel_size = [conv_kt[layer_no],conv_kf[layer_no]], 
												stride = [conv_st[layer_no],conv_sf[layer_no]], 
												w_scale_l1=0,
												w_scale_l2=0,
												b_scale_l1=0,
												b_scale_l2=0,
												sc='conv_ds_{}'.format(layer_no)
												)
	

						t_dim = math.ceil(t_dim/float(conv_st[layer_no]))
						f_dim = math.ceil(f_dim/float(conv_sf[layer_no]))

					net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')
				net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
		return net
		
	def _build_network_output(self):
	
		label_count = self._input_info['num_cls']

		#Extract model dimensions from model_size_info
		subnet_num = len(self._model_info["model_size_infos"])
		outs = []
		for id in range(subnet_num):
# 		scope = 'DS-CNN'
			net = self._gen_dscnn_structure(id)
			net = tf.layers.flatten(net)
			outs.append(net)
		net = tf.concat(outs, axis=-1)
		
#  		net = slim.fully_connected(net, label_count * 2, activation_fn=tf.nn.sigmoid, scope='fc01')
# 			net = slim.fully_connected(net, label_count * 30, activation_fn=tf.nn.relu, scope='fc02')
		logits = slim.fully_connected(net, label_count, activation_fn=None, scope='fc1')
		predicted_indices = tf.argmax(logits, 1)
		return logits, predicted_indices
		
