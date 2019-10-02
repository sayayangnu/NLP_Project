#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 01:07:35 2019

@author: Ivy
"""

import pandas as pd
import numpy as np
import pprint as pp
import preprocessor as p
import re
from nltk.util import ngrams
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords 
from collections import Counter
import matplotlib.pyplot as plt


def load_data(data):
    # Read in the txt file and turn into a pandas df 
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

def smart_preprocessing(text, hashtag=0, emoji=0, smiley=0, lemma=0, stop=0, bow=0, nut=0):
    # this function takes a piece of text and return the cleaned text in list format
    text_list = []
    # remove punctuations
    p.set_options(p.OPT.URL, p.OPT.MENTION)
    text = p.clean(text)
    if hashtag==1:
        p.set_options(p.OPT.HASHTAG)
        text = p.clean(text)
        # do we add hashparser ???
    if emoji==1:
        p.set_options(p.OPT.EMOJI)
        text = p.clean(text)
    if smiley==1:
        p.set_options(p.OPT.SMILEY)
        text = p.clean(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text_list = text.lower().split()
    #-------------------------
    if stop==1:
        text_list2 = []
        stop_words = set(stopwords.words('english'))
        # REMOVE or ADD the stop words you don't want here
        for w in text_list:
            if w not in stop_words:
                text_list2.append(w)
        text_list = text_list2
    if lemma==1:
        text_list2 = []
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        for w in text_list:
            # lemma might not work for putting, thats wierd
            stem = stemmer.stem(w)
            text_list2.append(stem)
        text_list = text_list2
    #--------------------------
    ## bag of word
    ## nut (not)
    
            
    return text_list
