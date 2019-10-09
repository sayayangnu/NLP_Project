#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:57:04 2019

@author: Tianyi Yang
"""

#from logistic_trial.py import *
from preprocessing import *
from  NaiveBayes import *

## Load train and test data
train_df = load_data('./P1_Data/Gold/merged.txt')
test_df = load_data('./P1_Data/Gold/test.txt') 
############ Note: We need to double check which dataset we use for Test ############


## List of parameters we plan to test 
test_param = [{"hashtag":1, "emo":0, "smiley":0, "stem":1, "stop":1, "nut":1}, # delete emojis
               {"hashtag":0, "emo":1, "smiley":1, "stem":1, "stop":1, "nut":1}, # keep hastag
               {"hashtag":1, "emo":1, "smiley":1, "stem":1, "stop":1, "nut":0}, # no negation
               {"hashtag":1, "emo":1, "smiley":1, "stem":1, "stop":1, "nut":1}]
############ Note: We need to add more test param options to find the best one ############

########################### TASK 3 ###########################

def get_best_nb_param(test_params):
    ## Among the test parameters provided above, find the best one for NB model
    params = []
    accuracy= []
    for i in range(len((test_params))):
        param = test_params[i]
        df= smart_preprocessing2(df=d1[:100], **param)
        ############ Note: NOW THIS IS ONLY d1[:100] AS SAMPLE CODE ############
        accuracy.append(nb_cv(df))
        params.append(param)
    result = pd.DataFrame({'Params':params, "Accuracy":accuracy})
    best_nb_param = result['Params'][result['Accuracy'].idxmax()]
    best_accuracy = result['Accuracy'].max()
    return best_nb_param,best_accuracy


def performance_best_nb(test_params,train_df,test_df):
    ## This returns the performance on the TEST set of the best NB model 
    best_nb_param = get_best_nb_param(test_params)[0]
    best_nb_df = smart_preprocessing2(train_df, **best_nb_param)
    sentiment_analyzer = SentimentAnalyzer()
    trainer = NaiveBayesClassifier.train
    classifier = sentiment_analyzer.train(trainer, get_nb_features(best_nb_df)) 
    truth_list = get_nb_features(test_df)      
    performance = sentiment_analyzer.evaluate(truth_list,classifier)
    results = {'Model':'Naive Bayes', 
               'F-measure[negative]': performance['F-measure [negative]'],
               'F-measure [neutral]': performance['F-measure [neutral]'],
               'F-measure [positive]': performance['F-measure [positive]'],
               'Avg Recall': (performance['Recall [negative]']+ 
                              performance['Recall [neutral]']+
                              performance['Recall [positive]'])/3
               }
    return results
   
    
## Run below to get performance 
## performance_best_nb(test_param,train_df,test_df)
'''
Result: 
{'Avg Recall': 0.6401315515008745,
 'F-measure [neutral]': 0.6741514900662251,
 'F-measure [positive]': 0.6708139611027438,
 'F-measure[negative]': 0.5476369092273069,
 'Model': 'Naive Bayes'}
'''

############ Note: We need the same thing for Logistic Regression ############
    
    
    


