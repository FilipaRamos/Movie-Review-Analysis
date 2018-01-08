import pandas as pd
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv1D, Flatten, MaxPooling1D, Embedding
from keras import optimizers, regularizers
from keras.optimizers import Nadam
import keras.utils
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

def preprocess(sentence):
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

X_clean = []
for review in X:
    X_clean.append(preprocess(review))

# Number of unique words 
print("Number of words: ", len(np.unique(np.hstack(X_clean))))

# Summarize review length
print("Review length: ")
result = [len(x) for x in X_clean]
print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))

# plot review length
pyplot.boxplot(result)
pyplot.show()


### Model definition ###

train_data, validation_data, train_labels, validation_labels = train_test_split(X_clean, y, train_size=0.8, test_size=0.2)

model = Sequential()
model.add(Embedding(input_dim=len(train_data), output_dim=32, input_length=40))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())


# Fitting the model: not working for now, some dimentionality issue in the Embedding 

#train_data, train_labels = shuffle(train_data, train_labels)
#model.fit(train_data, train_labels, validation_data=(validation_data, validation_labels), epochs=2, batch_size=128, verbose=2)
#print("accuracy of neural network :")
#err = model.evaluate(validation_data,validation_labels)
#print(err)