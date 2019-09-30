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
    else:
        return 0
'''
def train_logit(doc):
    cleaned_text = [clean_tweet(i) for i in doc['text']]
    words = [j for i in cleaned_text for j in i.split() if j not in stop_words]
    # word_counter -> other
    
    # generate vocabulary
    vocabulary = sorted(list(set(words)))
    # count = Counter(words)    
    # clean
    doc['text'] = doc['text'].apply(clean_tweet)
    x = doc['text'].apply(lambda x : bagofwords(x, vocabulary))
    X = np.vstack([i for i in x])
    y = doc.label.apply(y_ize)
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X,y)
    # performance on training set
    pre = logisticRegr.predict(X)
    # accuracy
    acc_train = sum(pre ==y)/len(y)
    print('training_accuracy = {}%'.format(str(round(acc_train,2)*100)))

    return logisticRegr
'''

'''
Bag of word code:
'''
doc_path = '/Users/chenlimin/Desktop/GU course/term3/nlp/project/data/Gold/train.txt'

# set pandas  for a pretty display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

stop_words = set(stopwords.words('english')) 

doc = load_data(doc_path)
#model = train_logit(doc)

# train the model
cleaned_text = [clean_tweet(i) for i in doc['text']]
words = [j for i in cleaned_text for j in i.split() if j not in stop_words]
# word_counter -> other

# generate vocabulary
vocabulary = sorted(list(set(words)))
# count = Counter(words)    
# clean
doc['text'] = doc['text'].apply(clean_tweet)
x = doc['text'].apply(lambda x : bagofwords(x, vocabulary))
X = np.vstack([i for i in x])
y = doc.label.apply(y_ize)
logisticRegr = LogisticRegression()
logisticRegr.fit(X,y)
# performance on training set
pre = logisticRegr.predict(X)
# accuracy
acc_train = sum(pre ==y)/len(y)
print('training_accuracy = {}%'.format(str(round(acc_train,2)*100)))

#################### test the model on the testing set ####################

test_txt = load_data('/Users/chenlimin/Desktop/GU course/term3/nlp/project/data/Gold/test.txt')
test_txt['text'] = test_txt['text'].apply(clean_tweet)

test_x = test_txt['text'].apply(lambda x : bagofwords(x, vocabulary))
test_x = np.vstack([i for i in test_x])

# not run
test_y = test_txt.label.apply(y_ize)

test_pre = logisticRegr.predict(test_x)
# accuracy
acc_test = sum(test_pre ==test_y)/len(test_y)
print('testing_accuracy = {}%'.format(str(round(acc_test,2)*100)))


'''
问题 & 优化版：
0.一开始忘记lemonize
1.尝试去增加 don't, didn't 等
2.尝试去添加 #，emoji等词汇
3.对词序理解，如应用n-gram
4.看资料，学习贝叶斯分类
5.bert，transformer，rnn
'''



























