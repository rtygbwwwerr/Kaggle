from keras.optimizers import Adam
from keras.layers import Dropout, Flatten, Conv1D, Conv2D, MaxPool1D, MaxPool2D, BatchNormalization
from keras.layers import Input, Dense, Masking, Merge, Permute
from keras.models import Sequential
import numpy as np # linear algebra

def make_cnn1(x_shape, cls_num, trainable = True):
    
    model = Sequential()
    
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',
                     input_shape = (x_shape[0], x_shape[1], 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
    model.add(BatchNormalization())
#     model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPool2D(strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(cls_num, activation='softmax'))
    model.trainable = trainable
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model