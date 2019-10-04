#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:57:04 2019

@author: chenlimin
"""

import os

##os.chdir('/Users/chenlimin/Desktop/GU course/term3/nlp/project/x11')


#from logistic_trial.py import *
from preprocessing import *
from  NaiveBayes import *


d1 = load_data('./P1_Data/Gold/merged.txt')

test_param = [{"hashtag":1, "emo":0, "smiley":0, "stem":1, "stop":1, "nut":1}, # delete emojis
               {"hashtag":0, "emo":1, "smiley":1, "stem":1, "stop":1, "nut":1}, # keep hastag
               {"hashtag":1, "emo":1, "smiley":1, "stem":1, "stop":1, "nut":0}, # no negation
               {"hashtag":1, "emo":1, "smiley":1, "stem":1, "stop":1, "nut":1}]

def get_best_param(test_params):
    params = []
    accuracy= []
    for i in range(len((test_params))):
        param = test_params[i]
        df= smart_preprocessing2(df=d1[:500], **param)
        ### NOW THIS IS ONLY d1[:500] AS SAMPLE CODE #####
        accuracy.append(nb_cv(df))
        params.append(param)
    result = pd.DataFrame({'Params':params, "Accuracy":accuracy})
    best_param = result['Params'][result['Accuracy'].idxmax()]
    best_accuracy = result['Accuracy'].max()
    return best_param,best_accuracy


    

    
    
    
    


