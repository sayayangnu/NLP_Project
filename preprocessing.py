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
import emoji

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

def smart_preprocessing(text, hashtag=0, emo=0, smiley=0, stem=0, stop=0, bow=0, nut=0):
    # this function takes a piece of text and return the cleaned text in list format
    text_list = []
    # remove punctuations
    p.set_options(p.OPT.URL, p.OPT.MENTION)
    text = p.clean(text)
    if hashtag==1:
        p.set_options(p.OPT.HASHTAG)
        text = p.clean(text)
        # do we add hashparser ???
    if emo==1:
        # p.set_options(p.OPT.EMOJI)
        # text = p.clean(text)
        text = emoji.demojize(text)
    if smiley==1:
        emoticon_dict = {":-)": "smiley",
                ":)": "smiley",
                ":]": "smiley",
                ":-]": "smiley",
                ":3": "smiley",
                ":-3": "smiley",
                ":-D": "laugh",
                ":D": "laugh",
                "X-D": "laugh",
                "XD": "laugh",
                "8D": "laugh",
                "8-D": "laugh",
                "<3": "love",
                ":-*": "kiss",
                ":*": "kiss",
                ":-(": "sad",
                ":(": "sad",
                ":-<": "sad",
                ":<": "sad",
                ":-C": "sad",
                ":C": "sad",
                ":'-(": "cry",
                ":'(": "cry",
                ":P": "tongue",
                ":-P": "tongue",
                "X-P": "tongue",
                "XP": "tongue",
                ";-)": "wink",
                ";)": "wink",
                "*-)": "wink",
                "*)": "wink",
                ";-]": "wink",
                ";]": "wink",
                ":/": "skeptical",
                "D‑':": "skeptical",
                "D:<": "disgust",
                "D:": "disgust",
                "D8": "disgust",
                "D;": "disgust"}
        for i, j in emoticon_dict.items():
            text = text.replace(i, j)
        #p.set_options(p.OPT.SMILEY)
        #text = p.clean(text)
    text = text.lower()
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
            '["\'?,\.]':'',
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
    ## nut (not)
    
            
    return text_list

def smart_preprocessing2(df, hashtag=0, emo=0, smiley=0, stem=0, stop=0, nut=0):
    transfer_col = df["text"].apply(lambda x: smart_preprocessing(x, hashtag, emo, smiley, stem, stop, nut))
    transfer_col_2 = transfer_col.apply(lambda x: ' '.join(i for i in x))
    df_new = df
    df_new['text']= transfer_col_2
    return df_new

if __name__ == "__main__":
    INPUT = load_data('P1_Data/Dev/INPUT.txt')
    
