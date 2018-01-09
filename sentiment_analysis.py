import pandas as pd
import numpy as np
from matplotlib import pyplot

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv1D, Flatten, MaxPooling1D, Embedding
from keras import optimizers, regularizers
#from keras.optimizers import Nadam
import keras.utils
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


### Load data ###

def get_review(file):
    with open(file) as file:
        review = file.readlines()
    review = [x.strip() for x in review] 
    return np.array(review)

neg = "data/rt-polarity-neg.txt"
X_neg = get_review(neg)
y_neg = np.zeros(X_neg.shape)

pos = "data/rt-polarity-pos.txt"
X_pos = get_review(pos)
y_pos = np.ones(X_pos.shape)

X = np.concatenate((X_neg,X_pos))
y = np.concatenate((y_neg,y_pos))


### Cleaning the reviews ###

# The data has been already cleaned up somewhat: 
# --> The dataset is comprised of only English reviews.
# --> All text has been converted to lowercase.
# --> There is white space around punctuation like periods, commas, and brackets.
# --> Text has been split into one sentence per line.

def process_row(sentence):
    '''
    Convert to lowercase
    Remove ponctuation 
    Remove unecessary words (stopwords)
    '''
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = list(filter(lambda token: token not in stopwords.words('english'), tokens))
    #Transform all plurals and conjugated verbs into simple form?
    return filtered_words

def process_data(data):
    data_clean = []
    for d in data:
        data_clean.append(process_row(d))
    
    vocab_size = len(np.unique(np.hstack(data_clean)))
    print("Number of words: ", vocab_size)
    
    result = [len(d) for d in data_clean]
    mean_length = np.mean(result)
    max_length = np.max(result)
    print("Mean %.2f words" % (mean_length))
    pyplot.boxplot(result)
    pyplot.show()
    
    encoded_data = [one_hot(" ".join(d), vocab_size) for d in data_clean]
    padded_data = pad_sequences(encoded_data, maxlen=max_length, padding='post')
    
    return vocab_size, mean_length, max_length, padded_data

vocab_size, mean_length, max_length, padded_data = process_data(X)

### Model definition ###

def neural_network():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model

model1 = neural_network()
model1.summary()

train_data, validation_data, train_labels, validation_labels = train_test_split(padded_data, y, train_size=0.8, test_size=0.2)

# Fit the model
train_data, train_labels = shuffle(train_data, train_labels)
model1.fit(train_data, train_labels, epochs=8, batch_size=64, verbose=1)

# Final evaluation of the model
loss, accuracy = model1.evaluate(validation_data, validation_labels, verbose=0)
print('\nLoss: %.2f' % (loss))
print('Accuracy: %.2f' % (accuracy*100))