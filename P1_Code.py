#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:42:14 2019

@author: apple
"""

import pandas as pd
import pprint as pp
import preprocessor as p
import re

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
    # Write a functino to clean emojis, hashtags, punctuations & urls
    clean_text = re.sub(r'[^\w\s]','',p.clean(text))
    return clean_text

## Data Exploration 
def explore_data(data):
    result = {}
    result['Num of Tweets']= len(data)
    result['Num of Characters']= sum(data['text'].apply(lambda x:len(x)))
    # Count Unique Vocab: clean all punctuations, hashtag, url and emojis
    all_vocabs=[]
    for text in data['text']:
        vocab = clean_tweet(text).split()
        all_vocabs.append(vocab)
    flat_vocab_list = [item for sublist in all_vocabs for item in sublist]
    result['Num of Unique Vocabs']= len(pd.unique(flat_vocab_list))
    # count avg. #characters  & # words
    result['Avg Character per Tweet']= round((data['text'].apply(lambda x:len(x))).mean(),2)
    result['Avg Word per Tweet']=round(data['text'].apply(lambda x: len(clean_tweet(x).split())).mean(),2)
    return result


def main():
## Read in the dev data set 
    dev= load_data('P1_Data/Dev/INPUT.txt')
    pp.pprint(explore_data(dev))

main()



