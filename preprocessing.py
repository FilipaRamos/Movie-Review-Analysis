import numpy as np

from keras.preprocessing.text import one_hot, hashing_trick
from keras.preprocessing.sequence import pad_sequences

from gensim.models import Word2Vec
from copy import deepcopy

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def onehot_encoding(data_clean, vocab_size, max_length):
    encoded_data = [one_hot(" ".join(d), vocab_size) for d in data_clean]
    padded_data = pad_sequences(encoded_data, maxlen=max_length, padding='post')
    return padded_data

def hash_encoding(data_clean, vocab_size, max_length):
    encoded_data = [hashing_trick(" ".join(d), vocab_size, hash_function='md5') for d in data_clean]
    padded_data = pad_sequences(encoded_data, maxlen=max_length, padding='post')
    return padded_data

def dict_freq(X_clean):
    word_list = np.hstack(X_clean)
    wordfreq = {}
    for word in word_list:
        if word not in wordfreq:
            wordfreq[word] = 0 
        wordfreq[word] += 1
    return wordfreq

def word2vec_embeding(X_clean, max_length, min_count=15, vector_size=100, window=5, negative=0, hs=1, workers=2, sg=1, iter=40):
    '''
    min_count: ignore all words with total frequency lower than this.
    size: is the dimensionality of the feature vectors.
    window: is the maximum distance between the current and predicted word within a sentence.
    negative: negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20).
    hs: hierarchical softmax will be used for model training.
    worker: use this many worker threads to train the model (=faster training with multicore machines).
    sg: defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
    iter: number of iterations (epochs) over the corpus. Default is 5.
    '''
    embedding = Word2Vec(X_clean, min_count=min_count, size=vector_size, window=window, negative=negative, hs=hs, workers=workers, sg=sg, iter=iter)
    X_vecs = embedding.wv

    wordfreq = dict_freq(X_clean)
    X_filtered = deepcopy(X_clean)
    i = 0
    for sentence in X_clean:
        for word in sentence:
            if wordfreq[word] < min_count+1:
                X_filtered[i].remove(word)
        i+=1 
    
    data_size = len(X_filtered)
    indexes = set(np.random.choice(data_size, data_size, replace=False))
    X_process = np.zeros((data_size, max_length, vector_size))
    for i, index in enumerate(indexes):
        for t, token in enumerate(X_filtered[index]):
            X_process[i, t, :] = X_vecs[token]
    
    return X_process