# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:54:00 2019

@author: rbshu
"""

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

analyzer = SentimentIntensityAnalyzer()

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


df_test = load_data('P1_data/Gold/devtest.txt')


def vader(data):
    vs = []
    for i in range(len(df_test)):
        vs_i = analyzer.polarity_scores(df_test['text'][i])
        vs.append(vs_i)
    #print(vs)
    
    compound_list = []
    label_list = []
    for vs_i in vs:
        compound = vs_i['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound > -0.05:
            label = 'neutral'
        else:
            label = 'negative'
            
        compound_list.append(compound)
        label_list.append(label)
    #print(label_list)
    
    result = pd.DataFrame(label_list,df_test['id'])
    result.columns = ['label']
    return result


def evaluate(testset):
    accuracy_list = []
    for i in range(len(testset)):
        if testset['label'][i] == result['label'][i]:
            accuracy = True
        else:
            accuracy = False
        accuracy_list.append(accuracy)
        accuracy_rate = sum(accuracy_list)/len(accuracy_list)
    return accuracy_rate



if __name__ == "__main__":
    df_train = load_data('P1_data/Gold/devtest.txt')
    result = vader(df_train)
    df_test = load_data('P1_data/Gold/devtest.txt')
    evaluation = evaluate(df_test)
    