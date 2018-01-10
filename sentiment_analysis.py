import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv1D, Flatten, MaxPooling1D, Embedding
from keras import optimizers, regularizers
#from keras.optimizers import Nadam
import keras.utils

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import preprocessing as pr

# random seed
seed = 7
np.random.seed(seed)

# get data
neg = "data/rt-polarity-neg.txt"
X_neg = pr.get_review(neg)
y_neg = np.zeros(X_neg.shape)

pos = "data/rt-polarity-pos.txt"
X_pos = pr.get_review(pos)
y_pos = np.ones(X_pos.shape)

X = np.concatenate((X_neg,X_pos))
y = np.concatenate((y_neg,y_pos))

vocab_size, mean_length, max_length, padded_data = pr.process_data(X)

def train_model(model, epochs, train_data, validation_data, train_labels, validation_labels):
    params_train = []
    params_valid = []
    for i in range(epochs):
        print('Epoch n°' + str(i+1))
        train_data, train_labels = shuffle(train_data, train_labels)
        model.fit(train_data, train_labels, epochs=1, batch_size=64, verbose = 2)
        print("accuracy of neural network :")
        err_train = model.evaluate(train_data, train_labels)
        err_valid = model.evaluate(validation_data, validation_labels)
        print(err_valid)
        params_train.append(err_train)
        params_valid.append(err_valid)
    return params_train, params_valid

def plot_params(params_train, params_valid):
    plt.plot(np.array(params_train).T[1],label='Train', C='C0')
    plt.plot(np.array(params_valid).T[1],label='Validation', C='C5', alpha=0.8)
    plt.legend(loc='best')
    plt.title('Accuracy across iterations')
    plt.show()
    
    plt.plot(np.array(params_train).T[0],label='Train', C='C0')
    plt.plot(np.array(params_valid).T[0],label='Validation', C='C5', alpha=0.8)
    plt.legend(loc='best')
    plt.title('Loss across iterations')
    plt.show()

###############################################################################################

### Multilayer perceptron ###

def mlp_network():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(25, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model

mlp_model = mlp_network()
mlp_model.summary()

# Fit the model
epochs = 5
model = mlp_model
mlp_train_data, mlp_validation_data, mlp_train_labels, mlp_validation_labels = train_test_split(padded_data, y, train_size=0.8, test_size=0.2)
params_train_mlp, params_valid_mlp = train_model(model, epochs, mlp_train_data, mlp_validation_data, mlp_train_labels, mlp_validation_labels)

# Evaluation of the model
plot_params(params_train_mlp, params_valid_mlp)

##############################################################################################

### Convolutional neural network definition ###

def conv_network():
    # create the model
    model = Sequential()
    
    # word embedding into vectors of numbers
    model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# create convolutional model
conv_model = conv_network()
conv_model.summary()

# Fit the model
epochs = 5
model = conv_model
conv_train_data, conv_validation_data, conv_train_labels, conv_validation_labels = train_test_split(padded_data, y, train_size=0.8, test_size=0.2)
params_train_conv, params_valid_conv = train_model(model, epochs, conv_train_data, conv_validation_data, conv_train_labels, conv_validation_labels)

# Final evaluation of the model
plot_params(params_train_conv, params_valid_conv)

