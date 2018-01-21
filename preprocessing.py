import numpy as np

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from gensim.models import Word2Vec

def onehot_encoding(data_clean, vocab_size, max_length):
    encoded_data = [one_hot(" ".join(d), vocab_size) for d in data_clean]
    padded_data = pad_sequences(encoded_data, maxlen=max_length, padding='post')
    return padded_data

def hash_encoding(data_clean, vocab_size, max_length):
    encoded_data = [hashing_trick(" ".join(d), vocab_size, hash_function='md5') for d in data_clean]
    padded_data = pad_sequences(encoded_data, maxlen=max_length, padding='post')
    return padded_data

def word2vec_embeding(data_clean, max_length, min_count, size, window, sg):
    '''
    min_count: words with an occurrence less than this count will be ignored
    size:  number of dimensions of the embeddin
    window: maximum distance between a target word and words around the target word
    sg: 0 CBOW, 1 skipgram
    '''
    embedding = Word2Vec(data_clean, min_count, size, window, sg)
    X_vecs = embedding.wv

    # Making all the input the same lenght: max_lenght
    data_length = len(data_clean)
    indexes = set(np.random.choice(data_length, data_length, replace=False))
    X_process = np.zeros((data_length, max_length, size))
    for i, index in enumerate(indexes):
        for t, token in enumerate(data_clean[index]):
            X_process[i, t, :] = X_vecs[token]

    return embedding, X_process