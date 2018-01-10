import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences


### Load data ###

def get_review(file):
    with open(file) as file:
        review = file.readlines()
    review = [x.strip() for x in review] 
    return np.array(review)


### Cleaning the reviews ###

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
    plt.boxplot(result)
    plt.show()
    
    encoded_data = [one_hot(" ".join(d), vocab_size) for d in data_clean]
    padded_data = pad_sequences(encoded_data, maxlen=max_length, padding='post')
    
    return vocab_size, mean_length, max_length, padded_data