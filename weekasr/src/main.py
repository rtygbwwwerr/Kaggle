import pandas as pd
import data_process
import model_maker_keras
import numpy as np
from config import Config as cfg
from keras.callbacks import LearningRateScheduler, EarlyStopping,TensorBoard, ReduceLROnPlateau, ModelCheckpoint
import time
import math
import os


def run_submission(filepath='../checkpoints/keras_cnn7_weights.41-0.0868-0.9718-0.2740-0.9343.hdf5'):
	x = data_process.get_test_data_from_files()
	x = np.reshape(x, x.shape + (1,))
	model = model_maker_keras.make_cnn1(x[0].shape, cfg.CLS_NUM, trainable=False)
	model.load_weights(filepath)
	submission(model, x)

def submission(model, x, file = '../data/submission.csv'):
	
	paths = data_process.gen_input_paths("../data/test/audio/", ".wav")
	fnames = map(lambda f:os.path.basename(f), paths)
	
	predict_y = model.predict(x, batch_size = 256, verbose=0)
	p_y = np.argmax(predict_y, axis=1)
	p_labels = map(lambda y:cfg.i2n(y), p_y)
	
	df_sub = pd.DataFrame({'fname':fnames, 'label':p_labels})
	print df_sub['label'].value_counts().sort_values()
	print "Submission samples:%d, file:%s"%(len(fnames), file)
	df_sub.to_csv(file, index=False)

def classify_generator(X, y, batch_size=128, shuffle=True):
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
# 		X_batch = np.zeros((len(batch_index), X.shape[1], cfg.input_classify_vocab_size))
		X_batch = X[batch_index,:]
		y_batch = y[batch_index,:]
		# reshape X to be [samples, time steps, features]
# 		for i, j in enumerate(batch_index):
# 			tmpx = X[j]
# 			for t in range(X.shape[1]):
#  				X_batch[i,t,tmpx[t]] = 1.0

		counter += 1
		yield X_batch, y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0

def train_keras_model(model, ret_file_head, X_train, Y_train, X_valid, Y_valid, batch_size=128, nb_epoch = 3):
	

	
	board = TensorBoard(log_dir='../logs/', histogram_freq=0, write_graph=True,
				 write_images=True, embeddings_freq=0, 
				 embeddings_layer_names=None, embeddings_metadata=None)
	check_file = "../checkpoints/%s_weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"%(ret_file_head)
	checkpointer = ModelCheckpoint(monitor="acc", filepath=check_file, verbose=1, save_best_only=True)
	# start training
	start_time = time.time()
 	
	samples_per_epoch = int(math.ceil(X_train.shape[0] / float(batch_size)))
# 	samples_per_epoch = batch_size * 2
	model.fit_generator(generator=classify_generator(X_train, Y_train, batch_size, True), 
	                    samples_per_epoch = samples_per_epoch, 
	                    nb_epoch = nb_epoch, 
	                    verbose=1,
	                    validation_data = (X_valid, Y_valid),
# 			    	    validation_data=sparse_generator(X_valid, Y_valid, batch_size, False), 
# 			    	    nb_val_samples=int(math.ceil(X_valid.shape[0] / float(batch_size))),
			    	    callbacks=[board, checkpointer]
			    		)
	print 'Training time', time.time() - start_time
	# evaluate network
	score = model.evaluate(X_valid, Y_valid, batch_size)

# 	val = np.max(p_y, axis=2)
# 	print val
	print('Test logloss:', score)

def experiment_keras_cnn(batch_size=256, nb_epoch=100, input_num=0, file_head="keras_cnn7", pre_train_model_prefix=None):
	x_train, y_train = data_process.get_training_data_from_files("../data/train/")
	x_valid, y_valid = data_process.get_training_data_from_files("../data/valid/")
	if input_num > 0:
		x_train = x_train[:input_num]
		y_train = y_train[:input_num]
	np.savetxt("../data/Y_train.txt", y_train.astype(np.int32), '%d')
	
	print "train items num:{0}, valid items num:{1}".format(x_train.shape[0], x_valid.shape[0])
	
	#add new dimension for channels
	x_train=np.reshape(x_train,x_train.shape + (1,))
	x_valid=np.reshape(x_valid,x_valid.shape + (1,))
# 	x_train = x_train[:, np.newaxis]
# 	x_valid = x_valid[:, np.newaxis]
	
	model = model_maker_keras.make_cnn1(x_train[0].shape, len(y_train[0]))
	print(model.summary())
	
	
	
	
	train_keras_model(model, file_head, x_train, y_train, x_valid, y_valid, batch_size, nb_epoch)







if __name__ == "__main__":
	run_submission()



