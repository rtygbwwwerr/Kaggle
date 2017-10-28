from keras import backend as K
from keras.engine.topology import Layer, InputSpec
import numpy as np
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec
from keras.layers import LSTM, activations, Wrapper, Recurrent
from keras.layers.recurrent import _time_distributed_dense
from keras.legacy import interfaces

class DropconnectDense(Layer):
    
    def __init__(self, units=100, rate=0.5,
                 noise_shape=None, seed=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DropconnectDense, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        
    def _get_noise_shape(self, _):
        return self.noise_shape
      
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        
        W = self.kernel
        output = K.dot(inputs, W)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
            
            def with_drop():
                W = K.dropout(self.kernel, self.rate, noise_shape, seed=self.seed)
                with_dropout = K.dot(inputs, W)
                if self.use_bias:
                    with_dropout = K.bias_add(with_dropout, self.bias)
                if self.activation is not None:
                    with_dropout = self.activation(with_dropout)
                print "dropout weights"
                return with_dropout
            #Only dropping in train phase 
            return K.in_train_phase(with_drop, output, training=training)
        
#         W = self.kernel
#         #Only dropping in train phase 
#         if K.learning_phase() == 1 :
#             W = K.dropout(self.kernel, self.rate, noise_shape, seed=self.seed)
#             print "dropout weights"
#         
#         print "phase" + str(K.learning_phase())
# 
#         output = K.dot(inputs, W)
#         if self.use_bias:
#             output = K.bias_add(output, self.bias)
#         if self.activation is not None:
#             output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


class AttentionLSTM(Recurrent):
	@interfaces.legacy_recurrent_support
	def __init__(self, units, attention_model,
				 activation='tanh',
				 recurrent_activation='hard_sigmoid',
				 use_bias=True,
				 kernel_initializer='glorot_uniform',
				 recurrent_initializer='orthogonal',
				 bias_initializer='zeros',
				 unit_forget_bias=True,
				 kernel_regularizer=None,
				 recurrent_regularizer=None,
				 bias_regularizer=None,
				 activity_regularizer=None,
				 kernel_constraint=None,
				 recurrent_constraint=None,
				 bias_constraint=None,
				 dropout=0.,
				 recurrent_dropout=0.,
				 **kwargs):
		super(LSTM, self).__init__(**kwargs)
		self.units = units
		self.activation = activations.get(activation)
		self.recurrent_activation = activations.get(recurrent_activation)
		self.use_bias = use_bias

		self.kernel_initializer = initializers.get(kernel_initializer)
		self.recurrent_initializer = initializers.get(recurrent_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.unit_forget_bias = unit_forget_bias

		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)

		self.kernel_constraint = constraints.get(kernel_constraint)
		self.recurrent_constraint = constraints.get(recurrent_constraint)
		self.bias_constraint = constraints.get(bias_constraint)

		self.dropout = min(1., max(0., dropout))
		self.recurrent_dropout = min(1., max(0., recurrent_dropout))
		self.state_spec = [InputSpec(shape=(None, self.units)),
						   InputSpec(shape=(None, self.units))]
		self.attention_model = attention_model
	def build(self, input_shape):
		if isinstance(input_shape, list):
			input_shape = input_shape[0]

		batch_size = input_shape[0] if self.stateful else None
		self.input_dim = input_shape[2]
		self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

		self.states = [None, None]
		if self.stateful:
			self.reset_states()

		self.kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
									  name='kernel',
									  initializer=self.kernel_initializer,
									  regularizer=self.kernel_regularizer,
									  constraint=self.kernel_constraint)
		self.recurrent_kernel = self.add_weight(
			shape=(self.units, self.units * 4),
			name='recurrent_kernel',
			initializer=self.recurrent_initializer,
			regularizer=self.recurrent_regularizer,
			constraint=self.recurrent_constraint)

		if self.use_bias:
			if self.unit_forget_bias:
				def bias_initializer(shape, *args, **kwargs):
					return K.concatenate([
						self.bias_initializer((self.units,), *args, **kwargs),
						initializers.Ones()((self.units,), *args, **kwargs),
						self.bias_initializer((self.units * 2,), *args, **kwargs),
					])
			else:
				bias_initializer = self.bias_initializer
			self.bias = self.add_weight(shape=(self.units * 4,),
										name='bias',
										initializer=bias_initializer,
										regularizer=self.bias_regularizer,
										constraint=self.bias_constraint)
		else:
			self.bias = None

		self.kernel_i = self.kernel[:, :self.units]
		self.kernel_f = self.kernel[:, self.units: self.units * 2]
		self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
		self.kernel_o = self.kernel[:, self.units * 3:]

		self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
		self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
		self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
		self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

		if self.use_bias:
			self.bias_i = self.bias[:self.units]
			self.bias_f = self.bias[self.units: self.units * 2]
			self.bias_c = self.bias[self.units * 2: self.units * 3]
			self.bias_o = self.bias[self.units * 3:]
		else:
			self.bias_i = None
			self.bias_f = None
			self.bias_c = None
			self.bias_o = None
			
		self.attention_model.build()
		self.built = True
		
		

	def preprocess_input(self, inputs, training=None):
		if self.implementation == 0:
			input_shape = K.int_shape(inputs)
			input_dim = input_shape[2]
			timesteps = input_shape[1]

			x_i = _time_distributed_dense(inputs, self.kernel_i, self.bias_i,
										  self.dropout, input_dim, self.units,
										  timesteps, training=training)
			x_f = _time_distributed_dense(inputs, self.kernel_f, self.bias_f,
										  self.dropout, input_dim, self.units,
										  timesteps, training=training)
			x_c = _time_distributed_dense(inputs, self.kernel_c, self.bias_c,
										  self.dropout, input_dim, self.units,
										  timesteps, training=training)
			x_o = _time_distributed_dense(inputs, self.kernel_o, self.bias_o,
										  self.dropout, input_dim, self.units,
										  timesteps, training=training)
			return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
		else:
			return inputs

	def get_constants(self, inputs, training=None):
		constants = []
		if self.implementation != 0 and 0 < self.dropout < 1:
			input_shape = K.int_shape(inputs)
			input_dim = input_shape[-1]
			ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
			ones = K.tile(ones, (1, int(input_dim)))

			def dropped_inputs():
				return K.dropout(ones, self.dropout)

			dp_mask = [K.in_train_phase(dropped_inputs,
										ones,
										training=training) for _ in range(4)]
			constants.append(dp_mask)
		else:
			constants.append([K.cast_to_floatx(1.) for _ in range(4)])

		if 0 < self.recurrent_dropout < 1:
			ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
			ones = K.tile(ones, (1, self.units))

			def dropped_inputs():
				return K.dropout(ones, self.recurrent_dropout)
			rec_dp_mask = [K.in_train_phase(dropped_inputs,
											ones,
											training=training) for _ in range(4)]
			constants.append(rec_dp_mask)
		else:
			constants.append([K.cast_to_floatx(1.) for _ in range(4)])
		return constants

	def step(self, inputs, states):
		h_tm1 = states[0]
		c_tm1 = states[1]
		dp_mask = states[2]
		rec_dp_mask = states[3]

		if self.implementation == 2:
			z = K.dot(inputs * dp_mask[0], self.kernel)
			z += K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel)
			if self.use_bias:
				z = K.bias_add(z, self.bias)

			z0 = z[:, :self.units]
			z1 = z[:, self.units: 2 * self.units]
			z2 = z[:, 2 * self.units: 3 * self.units]
			z3 = z[:, 3 * self.units:]

			i = self.recurrent_activation(z0)
			f = self.recurrent_activation(z1)
			c = f * c_tm1 + i * self.activation(z2)
			o = self.recurrent_activation(z3)
		else:
			if self.implementation == 0:
				x_i = inputs[:, :self.units]
				x_f = inputs[:, self.units: 2 * self.units]
				x_c = inputs[:, 2 * self.units: 3 * self.units]
				x_o = inputs[:, 3 * self.units:]
			elif self.implementation == 1:
				x_i = K.dot(inputs * dp_mask[0], self.kernel_i) + self.bias_i
				x_f = K.dot(inputs * dp_mask[1], self.kernel_f) + self.bias_f
				x_c = K.dot(inputs * dp_mask[2], self.kernel_c) + self.bias_c
				x_o = K.dot(inputs * dp_mask[3], self.kernel_o) + self.bias_o
			else:
				raise ValueError('Unknown `implementation` mode.')

			i = self.recurrent_activation(x_i + K.dot(h_tm1 * rec_dp_mask[0],
													  self.recurrent_kernel_i))
			f = self.recurrent_activation(x_f + K.dot(h_tm1 * rec_dp_mask[1],
													  self.recurrent_kernel_f))
			c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * rec_dp_mask[2],
															self.recurrent_kernel_c))
			o = self.recurrent_activation(x_o + K.dot(h_tm1 * rec_dp_mask[3],
													  self.recurrent_kernel_o))
		h = o * self.activation(c)
		if 0 < self.dropout + self.recurrent_dropout:
			h._uses_learning_phase = True
		return h, [h, c]

	def get_config(self):
		config = {'units': self.units,
				  'activation': activations.serialize(self.activation),
				  'recurrent_activation': activations.serialize(self.recurrent_activation),
				  'use_bias': self.use_bias,
				  'kernel_initializer': initializers.serialize(self.kernel_initializer),
				  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
				  'bias_initializer': initializers.serialize(self.bias_initializer),
				  'unit_forget_bias': self.unit_forget_bias,
				  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
				  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
				  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
				  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
				  'kernel_constraint': constraints.serialize(self.kernel_constraint),
				  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
				  'bias_constraint': constraints.serialize(self.bias_constraint),
				  'dropout': self.dropout,
				  'recurrent_dropout': self.recurrent_dropout}
		base_config = super(LSTM, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

#maxout
class MaxoutDense(Layer):
	'''
	 Maxout Layer
	 Reference:
	 Theano: new features and speed improvements.(https://arxiv.org/pdf/1211.5590.pdf)
	 http://blog.csdn.net/hjimce/article/details/50414467
	'''
	# input x shape(nb_samples, input_dim)  
	# output y shape(nb_samples, output_dim)  
	input_ndim = 2  
	#nb_feature k
	def __init__(self, output_dim, nb_feature=4,  
                 init='glorot_uniform', initializations, weights=None,  
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,  
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):  
		self.output_dim = output_dim  
		self.nb_feature = nb_feature  
		self.init = initializations.get(init)  
		
		self.W_regularizer = regularizers.get(W_regularizer)  
		self.b_regularizer = regularizers.get(b_regularizer)  
		self.activity_regularizer = regularizers.get(activity_regularizer)  
		
		self.W_constraint = constraints.get(W_constraint)  
		self.b_constraint = constraints.get(b_constraint)  
		self.constraints = [self.W_constraint, self.b_constraint]  
		
		self.initial_weights = weights  
		self.input_dim = input_dim  
		if self.input_dim:  
		    kwargs['input_shape'] = (self.input_dim,)  
		self.input = K.placeholder(ndim=2)  
		super(MaxoutDense, self).__init__(**kwargs)  
    #init  
	def build(self):  
		input_dim = self.input_shape[1]  
		
		self.W = self.init((self.nb_feature, input_dim, self.output_dim))#nb_featurec=k。  
		self.b = K.zeros((self.nb_feature, self.output_dim))  
		
		self.params = [self.W, self.b]
		self.regularizers = []
		
		if self.W_regularizer:  
		    self.W_regularizer.set_param(self.W)  
		    self.regularizers.append(self.W_regularizer)  
		
		if self.b_regularizer:  
			self.b_regularizer.set_param(self.b)  
			self.regularizers.append(self.b_regularizer)
		
		if self.activity_regularizer:  
		    self.activity_regularizer.set_layer(self)  
		    self.regularizers.append(self.activity_regularizer)  
		
		if self.initial_weights is not None:  
		    self.set_weights(self.initial_weights)  
		    del self.initial_weights  
  
	def get_output(self, train=False):  
		X = self.get_input(train)#X.shape=(nsamples,input_num)   
		# -- don't need activation since it's just linear.  
		output = K.max(K.dot(X, self.W) + self.b, axis=1)#maxout activation function
		return output 