#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:57:04 2019

@author: chenlimin
"""

import os

os.chdir('/Users/chenlimin/Desktop/GU course/term3/nlp/project/x11')


#from logistic_trial.py import *
from preprocessing import *
from  NaiveBayes import *


d1 = load_data('./P1_Data/Gold/merged.txt')

d1.shape


d1.head(20)

d2 = smart_preprocessing2(d1, hashtag=1, emo=1, smiley=1, stem=1, stop=1, bow=1, nut=1)



