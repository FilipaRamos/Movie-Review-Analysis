import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv1D, Flatten, MaxPooling1D, Embedding
from keras import optimizers, regularizers
import keras.utils

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import cleaning as cl
import preprocessing as pr
import tools

# random seed
seed = 7
np.random.seed(seed)

## get data ##
neg = "data/rt-polarity-neg.txt"
X_neg = cl.get_review(neg)
y_neg = np.zeros(X_neg.shape)

pos = "data/rt-polarity-pos.txt"
X_pos = cl.get_review(pos)
y_pos = np.ones(X_pos.shape)

X = np.concatenate((X_neg,X_pos))
y = np.concatenate((y_neg,y_pos))

## clean data ##
customize_stopwords = [
    'movie', 'movies'
    , 'film', 'films'
    , 'character', 'characters'
    , 'make', 'makes', 'made'
    , 'feel', 'feels', 'felt'
    , 'seem', 'seems', 'seemed'
    , 'one'
]
X_clean = cl.process_data(X, 'without_custom', customize_stopwords)

vocab_size, mean_length, max_length, Q1_length, med_length, Q3_length = cl.data_stat(X_clean)
cl.word_cloud(X_clean[:int(len(X_clean)/2)], 'viridis')
cl.word_cloud(X_clean[int(len(X_clean)/2):], 'magma')

## preprocess data ##
# For one hot encoding
X_onehot = pr.onehot_encoding(X_clean, vocab_size, max_length)
# For hashing trick encoding
X_hash = pr.hash_encoding(X_clean, vocab_size, max_length)
# For Word2Vec Embedding
word2vec_model, X_wor2vec = pr.word2vec_embeding(X_clean, max_length, min_count=1, size=200, window=3, sg=0)

## Split dataset 
def split_test(X,y):
    data = X
    data, y = shuffle(data, y)
    train_data, test_data, train_labels, test_labels = train_test_split(data, y, train_size=0.8, test_size=0.2)
    return train_data, test_data, train_labels, test_labels

train_data_oh, test_data_oh, train_labels_oh, test_labels_oh = split_test(X_onehot, y)
train_data_w2v, test_data_w2v, train_labels_w2v, test_labels_w2v = split_test(X_wor2vec, y)

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

##############################################################################################

### Convolutional neural network definition ###

def conv_network(encoding, vocab_size, max_length, vector_size=None):
    '''
    encoding: 'one-hot' or 'word2vec'
    vector_size: size of the word2vec embedding
    '''

    #create the model
    model = Sequential()
    
    if encoding == 'one-hot':
        model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length))
        model.add(BatchNormalization())
        model.add(Dropout(0.70))
        model.add(Conv1D(filters=64, kernel_size=3
                         , padding="same"
                         , activation='relu'))
    
    elif encoding == 'word2vec':
        model.add(Conv1D(filters=64, kernel_size=3
                         , padding="same"
                         , activation='relu'
                         , input_shape=(max_length, vector_size)))

    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Dense(50
                    , activation="relu"
                    , kernel_regularizer=regularizers.l2(0.00001)
                    , activity_regularizer=regularizers.l1(0.00001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model

## create convolutional model ##

model = conv_network('one-hot', vocab_size, max_length)
#model_w2v = conv_network('word2vec', vocab_size, max_length, 200)
conv_model.summary()

## Fit the model ##

# create an instance of accuracy history
history = tools.AccuracyLossHistory((test_data_oh, test_labels_oh))
#history_w2v = tools.AccuracyLossHistory((test_data_w2v, test_labels_w2v))

train_data_oh, train_labels_oh = shuffle(train_data_oh, train_labels_oh)
#train_data_w2v, train_labels_w2v = shuffle(train_data_w2v, train_labels_w2v)
model.fit(train_data_oh, train_labels_oh, epochs=10, batch_size=64, verbose=2, callbacks=[history])
#model_w2v.fit(train_data_w2v, train_labels_w2v, epochs=100, batch_size=64, verbose = 2, callbacks=[history_w2v])

# Final evaluation of the model
tools.plot_params(history.train, history.val)
#tools.plot_params(history_w2v.train, history_w2v.val)

##############################################################################################

### Recurrent neural network definition ###

def lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length))
    model.add(LSTM(28, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adagrad', metrics=["accuracy"])
    return model
