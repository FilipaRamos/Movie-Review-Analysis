import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv1D, Flatten, MaxPooling1D, Embedding
from keras import optimizers, regularizers
from keras.callbacks import ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import cleaning as cl
import preprocessing as pr
import tools 

# random seed
seed = 7
np.random.seed(seed)

# get data
neg = "data/rt-polarity-neg.txt"
X_neg = cl.get_review(neg)
y_neg = np.zeros(X_neg.shape)

pos = "data/rt-polarity-pos.txt"
X_pos = cl.get_review(pos)
y_pos = np.ones(X_pos.shape)

X = np.concatenate((X_neg,X_pos))
y = np.concatenate((y_neg,y_pos))

# clean data
# customize_stopwords = [
#     'movie', 'movies'
#     , 'film', 'films'
#     , 'character', 'characters'
#     , 'make', 'makes', 'made'
#     , 'feel', 'feels', 'felt'
#     , 'seem', 'seems', 'seemed'
#     , 'one'
# ]
X, y = shuffle(X, y)
X_clean = cl.process_data(X, 'with_stopwords')
vocab_size, mean_length, max_length, Q1_length, med_length, Q3_length = cl.data_stat(X_clean)

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

### Convolutional neural network ###

def model_to_complile(encoding, vocab_size, max_length, vector_size=0):
    '''
    encoding: 'discrete' or 'word2vec'
    vector_size: size of the word2vec embedding
    '''

    #create the model
    model = Sequential()
    
    if encoding == 'discrete':
        model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length))
        model.add(BatchNormalization())
        model.add(Dropout(0.70))
        model.add(Conv1D(filters=64, kernel_size=3
                         , padding="same"
                         , activation='relu'))
    
    elif encoding == 'word2vec':
        model.add(BatchNormalization(input_shape=(max_length, vector_size)))
        #model.add(Dropout(0.3, input_shape=(max_length, vector_size)))
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
    #model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model

##############################################################################################

### Recurrent neural network ###

def lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length))
    model.add(LSTM(28, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adagrad', metrics=["accuracy"])
    return model

##############################################################################################

### Train models and get results ###

def train_model(X_encoded, y, vocab_size, max_length, filepath, method, epochs, vector_size=0):
    '''
    method: 'discrete' or 'word2vec'
    '''
    # split dataset
    train_data, test_data, train_labels, test_labels = train_test_split(X_encoded, y, test_size=0.2, shuffle=False)
    # create model
    model = model_to_complile(method, vocab_size, max_length, vector_size)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # checkpoint and history 
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history = tools.AccuracyLossHistory((test_data, test_labels))
    # fit the model
    model.fit(train_data, train_labels, validation_split=0.2, epochs=epochs, batch_size=64, callbacks=[history, checkpoint], verbose=0)
    
    return history, [train_data, test_data, train_labels, test_labels]

def inverse_to_categorical(array):
    res = []
    for i in range(len(array)):
        if array[i] > 0.5:
            res.append(1)
        else:
            res.append(0)
    return res

def apply_best_model(data, filepath, method, vocab_size, max_length, vector_size=0):
    # create model
    model = model_to_complile(method, vocab_size, max_length, vector_size)
    # load weights
    model.load_weights(filepath)
    # compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("Created model and loaded weights from file")

    # estimate accuracy on the test dataset using loaded weights
    scores = model.evaluate(data[1], data[3], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.4f" % (model.metrics_names[0], scores[0]))
    # predict test labels and print confusion matrix
    yprob = model.predict(data[1])
    ypred = inverse_to_categorical(yprob)
    print('\nConfusion Matrix:\n', np.round(confusion_matrix(data[3], ypred)*100/len(data[3]),2))
    
    return scores, ypred

# One-hot encoding

X_onehot = pr.onehot_encoding(X_clean, vocab_size, max_length)
filepath_oh = "weights-improvement-oh.hdf5"
method_oh = 'discrete'
epochs_oh = 15

history_oh, data_oh = train_model(X_onehot, y, vocab_size, max_length, filepath_oh, method_oh, epochs_oh)
tools.plot_params(history_oh.train, history_oh.val)
scores_oh, ypred_oh = apply_best_model(data_oh, filepath_oh, method_oh, vocab_size, max_length)

# Hashing encoding

X_hash = pr.hash_encoding(X_clean, vocab_size, max_length)
filepath_hs = "weights-improvement-hs.hdf5"
method_hs = 'discrete'
epochs_hs = 15

history_hs, data_hs = train_model(X_onehot, y, vocab_size, max_length, filepath_hs, method_hs, epochs_hs)
tools.plot_params(history_hs.train, history_hs.val)
scores_hs, ypred_hs = apply_best_model(data_hs, filepath_hs, method_hs, vocab_size, max_length)

# Word2Vec embedding (skipgram)

min_count = 15
vector_size = 100
window = 5
negative = 0
hs = 1
sg = 1
workers = 2
iter = 40

X_word2vec = pr.word2vec_embeding(X_clean, max_length, min_count, vector_size, window, negative, hs, workers, sg, iter)
filepath_w2v = "weights-improvement-w2v.hdf5"
method_w2v = 'word2vec'
epochs_w2v = 15

history_w2v, data_w2v = train_model(X_word2vec, y, vocab_size, max_length, filepath_w2v, method_w2v, epochs_w2v, vector_size)
tools.plot_params(history_w2v.train, history_w2v.val)
scores_w2v, ypred_w2v = apply_best_model(data_w2v, filepath_w2v, method_w2v, vocab_size, max_length, vector_size)
