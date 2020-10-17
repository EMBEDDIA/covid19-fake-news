# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 8:27:03 2020

@author: Bosec
"""


import config
from parse_data import *
from statistical_features import *
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


train = readTrain()
data_matrix = build_features(train['tweet'])
print(data_matrix)
parameters = {"loss":["hinge","log"],"penalty":["elasticnet"],"alpha":[0.01,0.001,0.0001,0.0005],"l1_ratio":[0.05,0.25,0.3,0.6,0.8,0.95],"power_t":[0.5,0.1,0.9]}
x = SGDClassifier()
clf1 = GridSearchCV(x, parameters, verbose = 0, n_jobs = 8,cv = 10, refit = True)
scores = cross_val_score(clf1, data_matrix, train['label'], cv=10, scoring='f1_macro', verbose = True)
y = [1 if c  == 'real' else 0 for c in train['label'].to_list()]

clf1 = clf1.fit(data_matrix, y)

print("AVG F1-score:" + str(scores.mean()))
valid = readValidation()
y = [1 if c  == 'real' else 0 for c in valid['label'].to_list()]
data_matrix_val = build_features(valid['tweet'])
print(f1_score(clf1.predict(data_matrix_val), y))