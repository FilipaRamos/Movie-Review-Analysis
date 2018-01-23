import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from wordcloud import WordCloud

### Load data ###

def get_review(file):
    with open(file) as file:
        review = file.readlines()
    review = [x.strip() for x in review] 
    return np.array(review)


### Cleaning the reviews ###

def process_row(sentence, method='with_stopwords', customize_stopwords=None):
    '''
    Convert to lowercase
    Split into words removing ponctuation 
    Remove unecessary words (stopwords)
    '''
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    if method == 'with_stopwords':
        return tokens
    elif method in ('without_stopwords', 'without_custom'):
        filtered_words = list(filter(lambda token: token not in customize_stopwords, tokens))
        if method == 'without_custom':
            filtered_words = list(filter(lambda token: token not in stopwords.words('english'), filtered_words))
        return filtered_words
    
def process_data(data,method='with_stopwords', customize_stopwords=[]):
    data_clean = []
    for d in data:
        data_clean.append(process_row(d, method, customize_stopwords))   
    return data_clean

def data_stat(data_clean):
    vocab_size = len(np.unique(np.hstack(data_clean)))
    print("Number of words: ", vocab_size)
    
    result = [len(d) for d in data_clean]
    mean_length = np.mean(result)
    max_length = np.max(result)
    Q1_length = np.percentile(result, 25)
    med_length = np.percentile(result, 50)
    Q3_length = np.percentile(result, 75)
    print("Mean %.0f words" % (mean_length))
    print("Max %.0f words" % (max_length))
    print("Q1: %.0f words, Med: %.0f, Q3: %.0f" % (Q1_length, med_length, Q3_length))
    
    plt.boxplot(result)
    plt.show()

    return vocab_size, mean_length, max_length, Q1_length, med_length, Q3_length

### Word Cloud Visualization ###

def word_cloud(X_clean, colormap):
    X = [" ".join(d) for d in X_clean]
    string = []
    for t in X:
        string.append(t)
    string = pd.Series(string).str.cat(sep=' ')
    
    wordcloud = WordCloud(width=1600, height=800,max_font_size=200, colormap=colormap).generate(string)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


