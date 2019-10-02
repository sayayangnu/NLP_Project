#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:12:22 2019

@author: apple
"""

import pandas as pd
import numpy as np
import preprocessor as p
import re
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from sklearn.model_selection import KFold

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


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


'''
BELOW IS JUST FOR TIANYI'S RAW TESTING PURPOSE
We will need dynamic cleaning
We need to use a bigger dataset 
'''    
df_train=load_data('P1_data/Gold/train.txt') 
cleaned_df = df_train.iloc[:,:2]
cleaned_df['text']= [clean_tweet(x) for x in df_train['text']]
''' 
Test Code End Here 
'''


### THE MODEL ###

def features(sentence):
    words = sentence.lower().split()
    return dict(('contains(%s)' % w, True) for w in words)

def get_nb_features(cleaned_df):
    # Extract Positive, Negative, Netural Tweets 
    df_pos_train = cleaned_df[cleaned_df['label'] == 'positive']
    pos_tweets = df_pos_train['text'].tolist()
    
    df_neg_train = cleaned_df[cleaned_df['label'] == 'negative']
    neg_tweets = df_neg_train['text'].tolist()
    
    df_neutral_train = cleaned_df[cleaned_df['label'] == 'neutral']
    neutral_tweets = df_neutral_train['text'].tolist()
    # Create Traning Features 
    positive_featuresets = [(features(tweet),'positive') for tweet in pos_tweets]
    negative_featuresets = [(features(tweet),'negative') for tweet in neg_tweets]
    neutral_featuresets = [(features(tweet),'neutral') for tweet in neutral_tweets]
    training_features = positive_featuresets + negative_featuresets + neutral_featuresets
    
    return training_features

def nb_cv(cleaned_df):   
    # Get Features
    training_set = get_nb_features(cleaned_df)
    # Get 10-Fold. Important: Shuffle=True
    cv = KFold(n_splits=10, random_state=0, shuffle=True)
    # Model 
    sentiment_analyzer = SentimentAnalyzer()
    trainer = NaiveBayesClassifier.train
    # Store Result
    Accuracy = []
    # For each fold, train model, evaluate 
    for train_index, test_index in cv.split(training_set):
        classifier = sentiment_analyzer.train(trainer, np.array(training_set)[train_index].tolist())        
        truth_list = np.array(training_set)[test_index].tolist()       
        performance = sentiment_analyzer.evaluate(truth_list,classifier)
        Accuracy.append(performance['Accuracy'])
        '''## Can add all other measures here. Sample Result as below: 
        {'Accuracy': 0.525, 
        'Precision [negative]': 0.28337874659400547, 'Recall [negative]': 0.7272727272727273, 'F-measure [negative]': 0.407843137254902, 
        'Precision [neutral]': 0.5011933174224343, 'Recall [neutral]': 0.30837004405286345, 'F-measure [neutral]': 0.38181818181818183, 
        'Precision [positive]': 0.7461629279811098, 'Recall [positive]': 0.611810261374637, 'F-measure [positive]': 0.672340425531915}
        '''
    return np.mean(np.asarray(Accuracy))



nb_cv(cleaned_df)




















