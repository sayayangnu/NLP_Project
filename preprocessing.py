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

def smart_preprocessing(text, hashtag=0, emoji=0, smiley=0, stem=0, stop=0, bow=0, nut=0):
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
    text = text.lower()
    punctuation='["\'?,\.]' # I will replace all these punctuation with ''
    abbr_dict={"what's":"what is",
            "what're":"what are",
            "who's":"who is",
            "who're":"who are",
            "where's":"where is",
            "where're":"where are",
            "when's":"when is",
            "when're":"when are",
            "how's":"how is",
            "how're":"how are",

            "i'm":"i am",
            "we're":"we are",
            "you're":"you are",
            "they're":"they are",
            "it's":"it is",
            "he's":"he is",
            "she's":"she is",
            "that's":"that is",
            "there's":"there is",
            "there're":"there are",

            "i've":"i have",
            "we've":"we have",
            "you've":"you have",
            "they've":"they have",
            "who've":"who have",
            "would've":"would have",
            "not've":"not have",

            "i'll":"i will",
            "we'll":"we will",
            "you'll":"you will",
            "he'll":"he will",
            "she'll":"she will",
            "it'll":"it will",
            "they'll":"they will",

            "isn't":"is not",
            "wasn't":"was not",
            "aren't":"are not",
            "weren't":"were not",
            "can't":"can not",
            "couldn't":"could not",
            "don't":"do not",
            "didn't":"did not",
            "shouldn't":"should not",
            "wouldn't":"would not",
            "doesn't":"does not",
            "haven't":"have not",
            "hasn't":"has not",
            "hadn't":"had not",
            "won't":"will not",
            punctuation:'',
            '\s+':' ', # replace multi space with one single space
            }
    for i, j in abbr_dict.items():
        text = text.replace(i, j)
        
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text_list = text.split()
    #-------------------------
    if stop==1:
        text_list2 = []
        stop_words = set(stopwords.words('english'))
        # REMOVE or ADD the stop words you don't want here
        # stop words include all not words and but
        # we need to think of this 
        for w in text_list:
            if w not in stop_words:
                text_list2.append(w)
        text_list = text_list2
    if stem==1:
        text_list2 = []
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

if __name__ == "__main__":
    INPUT = load_data('P1_Data/Dev/INPUT.txt')
    
