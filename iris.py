# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 00:14:08 2022

@author: shree
"""

from fileinput import filename
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
#import pickle

iris = pd.read_csv('iris.csv')

iris.drop('Id', axis=1, inplace = True)

x = iris.drop('Species', axis=1)
y = iris['Species']

le = LabelEncoder()

y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)

model = SVC()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

#pickle.dump(model,open('iris.pkl','wb'))

filename = 'model.sav'
joblib.dump = (model, filename)