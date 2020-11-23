# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 8:27:03 2020

@author: Bosec
"""


import config
import parse_data
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
import pickle
import os
def train():
    train = parse_data.get_train()
    train_texts = train["text_a"].to_list()
    train_y = train["label"].to_list()
    data_matrix = build_features(train_texts)
    print("TRAINING: ")
    parameters = {"loss":["hinge","log"],"penalty":["elasticnet"],"alpha":[0.01,0.001,0.0001,0.0005],"l1_ratio":[0.05,0.25,0.3,0.6,0.8,0.95],"power_t":[0.5,0.1,0.9]}
    x = SGDClassifier()
    clf1 = GridSearchCV(x, parameters, n_jobs = 8,cv = 10, refit = True, verbose = 10)
    scores = cross_val_score(clf1, data_matrix, train_y, n_jobs = 8, cv=10, scoring='f1_macro', verbose = 10)
    clf1 = clf1.fit(data_matrix, train_y)
    score_train = f1_score(train_y, clf1.predict(data_matrix))
    
    dev = parse_data.get_dev()
    dev_features = build_features(dev['text_a'])
    dev_score = f1_score(dev["label"].to_list(), clf1.predict(dev_features))
    
    
    test = parse_data.get_test()
    test_features = build_features(test['text_a'])
    test_score = f1_score(test["label"].to_list(), clf1.predict(test_features))

def fit(texts):
    features = build_features(texts)
    with open(os.path.join(config.PICKLES_PATH,"clf_stat.pkl"), "rb") as f:
        model = pickle.load(f)
    predictions = model.predict(features)
    return predictions

def fit_probs(texts):
    features = build_features(texts)
    with open(os.path.join(config.PICKLES_PATH,"clf_stat.pkl"), "rb") as f:
        model = pickle.load(f)
    try:
        predictions = model.decision_function(features)
    except:
        predictions = model.predict_proba(features)
    return predictions
    
    
