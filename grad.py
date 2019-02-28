#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:44:41 2019

@author: ashutoshjoshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Admission_Predict_Ver1.1.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

from keras import initializers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler


model = Sequential()
model.add(Dense(10,input_dim = len(x_train[0]),kernel_initializer='random_uniform', activation='sigmoid' ))
model.add(Dropout(0.2))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, kernel_initializer='normal'))


model.compile(loss = 'mean_squared_error', optimizer= 'adam')

model.fit(x_train,y_train, validation_split=0.25, batch_size= 5, epochs=  100)


y_pred = model.predict(x_test)

from sklearn.metrics import mean_absolute_error
score = mean_absolute_error(y_test,y_pred)

print("The mean erro is = "+ str(score))

model.save('My_Model.h5')




 