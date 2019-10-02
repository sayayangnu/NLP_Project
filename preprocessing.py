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

def smart_preprocessing(text, hashtag=0, emoji=0, smiley=0, lemma=1, stop=0, bow=1, nut=0):
    # this function takes a piece of text and return the cleaned text in list format
    text_list = []
    p.set_options(p.OPT.URL, p.OPT.MENTION)
    # p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.EMOJI,p.OPT.SMILEY,p.OPT.HASHTAG)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', p.clean(text))
    if hashtag==0:
        p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', p.clean(text))
    if emoji==0:
        p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.EMOJI)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', p.clean(text))
    if smiley==0:
        p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.SMILEY)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', p.clean(text))
    for w in text:
        text_list.append(w)
    #-------------------------
    if stop==0:
        text_list = []
        stop_words = set(stopwords.words('english'))
        # REMOVE or ADD the stop words you don't want here
        for w in text:
            if w not in stop_words:
                text_list.append(w)
    if lemma==0:
        text_list_copy = text_list
        text_list = []
        lemmatizer = WordNetLemmatizer()
        for w in text_list_copy:
            lemma = lemmatizer.lemmatize(w)
            text_list.append(lemma)
    #--------------------------
    ## bag of word
    ## nut (not)
    
            
        
    
    
    return