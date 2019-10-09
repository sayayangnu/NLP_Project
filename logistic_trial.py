#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:27:00 2019

@author: chenlimin
"""

import pandas as pd
import numpy as np
import pprint as pp
import preprocessor as p
import re
from nltk.util import ngrams
from nltk.tokenize import TweetTokenizer
from collections import Counter 
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.corpus import stopwords 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from preprocessing import *

###### Task 1: Exploratory Data analysis (5 points) ######

def load_data(data):
    ## Read in the txt file and turn into a pandas df 
    ids = []
    labels = []
    texts = []
    with open(data, 'r') as f:
        for line in f:
            if line.strip():
                fields = line.lower().strip().split("\t")
                ids.append(fields[0])
                labels.append(fields[1])
                texts.append(fields[2])
        df = pd.DataFrame(
    {'id': ids,
     'label': labels,
     'text': texts
    })
    return df

def clean_tweet(text):
    # Write a function to clean emojis, smileys, mentions, punctuations & urls
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.EMOJI,p.OPT.SMILEY,p.OPT.HASHTAG)
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', p.clean(text))
    return clean_text

def extract_words(sentence):
    ignore_words = ['a']
    words = re.sub("[^\w]", " ",  sentence).split() #nltk.word_tokenize(sentence)
    words_cleaned = [w.lower() for w in words if w not in ignore_words]
    return words_cleaned 

def tokenize_sentences(sentences):
    words = []
    for sentence in sentences:
        w = extract_words(sentence)
        words.extend(w)
        
    words = sorted(list(set(words)))
    return words

# 这个代码会把 other 直接忽视，因为默认 other 对结果没有影响
def bagofwords(sentence, vocabulary):
    sentence_words = extract_words(sentence)
    # frequency word count
    bag = np.zeros(len(vocabulary))
    for sw in sentence_words:
        for i,word in enumerate(vocabulary):
            if word == sw: 
                bag[i] += 1
                
    return np.array(bag)

def y_ize(x):
    if x == 'positive':
        return 1
    elif x == 'negative':
        return -1
    else:
        return 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

stop_words = set(stopwords.words('english')) 

def train_logit(doc):
    
    words = [j for i in doc['text'] for j in i.split() if j not in stop_words]
    # word_counter -> other
    
    # generate vocabulary
    vocabulary = sorted(list(set(words)))
    # count = Counter(words)    
    # clean
    x = doc['text'].apply(lambda x : bagofwords(x, vocabulary))
    X = np.vstack([i for i in x])
    y = doc.label.apply(y_ize)
    
    train_x, test_x, train_y, test_y = train_test_split(X,y, train_size=0.7)

    logisticRegr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    logisticRegr.fit(train_x,train_y)
    # performance on training set
    pre = logisticRegr.predict(test_x)
    # accuracy
    acc_test = sum(pre ==test_y)/len(test_y)
    print('testing_accuracy = {}%'.format(str(round(acc_test,2)*100)))
    
    return logisticRegr


# logistic_model = train_logit(d2)


if __name__ == '__main__':
    d1 = load_data('P1_Data/Gold/merged.txt')
    d2 = smart_preprocessing2(d1)
    m1 = train_logit(d2)
    
    




































