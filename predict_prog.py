#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:47:41 2019

@author: ashutoshjoshi
"""

from keras.models import load_model
import numpy as np

model = load_model('My_Model.h5')

words = ['GRE Score',
         'TOEFL Score', 
         'University Ranking(1-5)',
         'SOP(1-5)',
         'LOR(1-5)',
         'CGPA',
         'Research(0/1)']

pred = np.zeros(shape = (1,7))

i = 0
for word in words:
    a = input("Enter the " + word + ": " )
    pred[0][i] = float(a)
    i = i+1
    
y_pred = model.predict(pred)    

print("The Probability of acceptance is : "+ str(y_pred[0][0]*100) + "%")