#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:42:14 2019

@author: apple, Ivy
"""

import pandas as pd
import numpy as np
import pprint as pp
import preprocessor as p
import re
import nltk
from nltk.tokenize import TweetTokenizer
from collections import Counter 

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
    # Write a function to clean emojis, hashtags, punctuations & urls
    clean_text = re.sub(r'[^\w\s]','',p.clean(text))
    return clean_text

## Data Exploration 
def input_data_explore(data):
    # Explore INPUT data
    result = {}
    # 1. Total Number of Tweets
    result['Num of Tweets'] = len(data)
    # 2. Total Number of Characters:
    result['Num of Char'] = sum(data['text'].apply(lambda x:len(x)))
    # 3. Total Number of Distinct Words (Vocalbulary)
    # All punctuations, hashtag, urls and emojis are cleaned
    all_vocab_list = []
    for t in data['text']:
        vocab = clean_tweet(t).split()
        all_vocab_list.append(vocab)
    all_vocab = [item for sublist in all_vocab_list for item in sublist]
    all_type = pd.unique(all_vocab)
    result['Num of Distinct Vocab'] = len(all_type)
    # 4.1. The Average Number of Characters in Each Tweet
    result['Avg Char per Tweet'] = round((data['text'].apply(lambda x:len(x))).mean(),2)
    # 4.2. The Average Number of Words in Each Tweet (Tweets Cleaned)
    result['Avg Word per Tweet'] = round(data['text'].apply(lambda x:len(clean_tweet(x).split())).mean(),2)
    # 5.1. The Average Number of Characters per Token
    all_token_list = []
    tt = TweetTokenizer()
    for t in data['text']:
        token = tt.tokenize(t)
        all_token_list.append(token)
    all_token = [item for sublist in all_token_list for item in sublist]
    result['Avg Char per Token'] = round(np.mean(list(map(lambda x:len(x), all_token))),2)
    # 5.2. The Standard Deviation of Characters per Token
    result['Std Char per Token'] = round(np.std(list(map(lambda x:len(x), all_token))),2)
    # 6. Number of Token for Each of Top 10 Most Frequent Words (Types)
    result['Top 10 Types and Frequency'] = Counter(all_vocab).most_common(10)
    # 7. Token / Type Ratio
    result['Token / Type Ratio'] = round(len(all_token) / len(all_type),2)
    # 8. Number of N-grams (of words) for n = 2,3,4,5
    
    # 9. Number of Distinct N-grams (of char) for n = 2,3,4,5,6,7
    
    # 10. Plot a token log frequency. 
    # Describe what this plot means and how to interpret it. 
    # Describe out it might help you understand coverage when training a model?
    
    return result

def main():
## Read in the INPUT data set and do explore data analysis (EDA) on it
    INPUT = load_data('P1_Data/Dev/INPUT.txt')
    pp.pprint(input_data_explore(INPUT))

main()



