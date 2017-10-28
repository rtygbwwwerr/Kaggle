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
import seq2seq

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
# 	    encoder.add(Dropout(dropout))
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
# 	        decoder.add(Dropout(dropout))
	        decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
# 	    decoder.add(Dropout(dropout))
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
	                    tf.ones_like(inputs) * vocab.word2id[u"UNK"])

# placeholder for inputs
encoder_inputs = tf.placeholder(tf.int32, shape=(cfg.batch_size, None))
decoder_inputs = tf.placeholder(tf.int32, shape=(cfg.batch_size, None))
decoder_outputs = tf.placeholder(tf.int32, shape=(cfg.batch_size, None))

# placeholder for sequence lengths
encoder_inputs_lengths = tf.placeholder(tf.int32, shape=(cfg.batch_size,))
decoder_inputs_lengths = tf.placeholder(tf.int32, shape=(cfg.batch_size,))

def make_tf_copyNet(mode='train'):
	assert mode in {'train', 'eval', 'infer'}, 'invalid mode!'
	# embedding maxtrix
	embedding_matrix = tf.get_variable("embedding_matrix", [
		cfg.vocab_size, cfg.embedding_size])
	encoder_inputs_emb = tf.nn.embedding_lookup(embedding_matrix, 
										replace_with_unk(encoder_inputs))
	decoder_inputs_emb = tf.nn.embedding_lookup(embedding_matrix, 
										replace_with_unk(decoder_inputs))

	# encoder
	encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(cfg.encoder_hidden_size)
	encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
								encoder_cell,
								encoder_inputs_emb,
								sequence_length=encoder_inputs_lengths,
								time_major=False,
								dtype=tf.float32)

	# attention wrapper for decoder_cell
	attention_mechanism = seq2seq.LuongAttention(
							cfg.decoder_hidden_size, 
							encoder_outputs, 
							memory_sequence_length=encoder_inputs_lengths)
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
					tf.fill([cfg.batch_size], vocab.word2id[u"START"]),
					vocab.word2id[u"END"])
		decoder = seq2seq.BasicDecoder(decoder_cell,
												  helper,
												  decoder_initial_state,
												  output_layer=projection_layer)
		maximum_iterations = tf.round(tf.reduce_max(encoder_inputs_lengths) * 2)
		final_outputs, final_state, seq_len = \
				seq2seq.dynamic_decode(decoder,
									maximum_iterations=maximum_iterations)
		translations = final_outputs.sample_id
		return translations

	# train or eval mode
	helper = seq2seq.CopyNetTrainingHelper(decoder_inputs_emb, encoder_inputs,
											   decoder_inputs_lengths, 
											   time_major=False)
	decoder = seq2seq.CopyNetDecoder(cfg, decoder_cell, helper,
											  decoder_initial_state,
											  encoder_outputs,
											  output_layer=projection_layer)
	final_outputs, final_state, seq_lens = \
						seq2seq.dynamic_decode(decoder)
	logits = final_outputs.rnn_output

	# loss
	crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=decoder_outputs, logits=logits)
	max_seq_len = logits.shape[1].value
	target_weights = tf.sequence_mask(decoder_inputs_lengths, max_seq_len, 
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