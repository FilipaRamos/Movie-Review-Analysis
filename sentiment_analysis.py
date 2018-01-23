import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv1D, Flatten, MaxPooling1D, Embedding
from keras import optimizers, regularizers

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

def conv_network(encoding, vocab_size, max_length, vector_size=0):
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
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
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

def train_embedding_model(X_encoded, y, vocab_size, max_length, method, epochs, vector_size=0):
    '''
    method: discrete or word2vec
    '''
    train_data, test_data, train_labels, test_labels = train_test_split(X_encoded, y
                                                                                , test_size=0.2, shuffle=False)
    model = conv_network(method, vocab_size, max_length, vector_size)
    history = tools.AccuracyLossHistory((test_data, test_labels))
    model.fit(train_data, train_labels, epochs=epochs, batch_size=64, verbose=2, callbacks=[history])
    
    tools.plot_params(history.train, history.val)
    
    index = np.argmax(np.asarray(history.val).T[1])
    print('** Best ** \nValidation loss %.2f \nValidation accuracy %.2f' % (history.val[index][0], history.val[index][1]*100))
    print('\n** Last ** \nValidation loss %.2f \nValidation accuracy %.2f' % (history.val[epochs-1][0], history.val[epochs-1][1]*100))
    
    return history, [train_data, test_data, train_labels, test_labels]

def inverse_to_categorical(array):
    res = []
    for i in range(len(array)):
        if array[i] > 0.5:
            res.append(1)
        else:
            res.append(0)
    return res

# One-hot encoding

X_onehot = pr.onehot_encoding(X_clean, vocab_size, max_length)
history_oh, data_oh = train_embedding_model(X_onehot, y, vocab_size, max_length, 'discrete', 12)

yprob_oh = history_oh.model.predict(data_oh[1])
ypred_oh = inverse_to_categorical(yprob_oh)
print('Confusion Matrix:\n', np.round(confusion_matrix(data_oh[3], ypred_oh)*100/len(data_oh[3]),2))

# Hashing encoding

X_hash = pr.hash_encoding(X_clean, vocab_size, max_length)
history_hs, data_hs = train_embedding_model(X_hash, y, vocab_size, max_length, 'discrete', 15)

yprob_hs = history_hs.model.predict(data_hs[1])
ypred_hs = inverse_to_categorical(yprob_hs)
print('Confusion Matrix:\n', np.round(confusion_matrix(data_hs[3], ypred_hs)*100/len(data_hs[3]),2))

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
history_w2v, data_w2v = train_embedding_model(X_word2vec, y, vocab_size, max_length, 'word2vec', 12, vector_size)

yprob_w2v = history_w2v.model.predict(data_w2v[1])
ypred_w2v = inverse_to_categorical(yprob_w2v)
print('Confusion Matrix:\n', np.round(confusion_matrix(data_w2v[3], ypred_w2v)*100/len(data_w2v[3]),2))
