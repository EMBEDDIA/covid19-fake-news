# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 8:27:03 2020

@author: Bosec
"""


import config
import logging
import parse_data
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import feature_construction
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


#BERT
class BERTTransformer():
    def __init__(self,docs, model_name = 'distilbert-base-nli-mean-tokens'):
        self.docs = list(docs.values())
        self.model_name = model_name
        self.docsNames = list(docs.keys())
        self._model = None
        self.X = None
        
    def fit(self):
        self._model = SentenceTransformer(self.model_name)
        self.X = self._model.encode(self.docs)  
   
    def to_matrix(self):
        return np.array(self.X)
    
    def _exportModel(self,path="bert_sentences.pkl"):
        pickle.dump(self.X, open(path, "wb"))
        
    def _importModel(self,path="bert_sentences.pkl"):
        self.X = pickle.load(open(path,'rb'))
        return self.X
    
    def _export(self,path="bert_transfomer.pkl"):
        pickle.dump(self._model, open(path, "wb"))

    def _import(self,path="bert_.pkl"):
        self._model = pickle.load(open(path,'rb'))
    

def preprocess_texts(docs):
    dataframe = feature_construction.build_dataframe(docs)
    return dataframe['no_stopwords']
    
def fit_space(texts, model = "distilbert-base-nli-mean-tokens"):
    preprocessed = preprocess_texts(texts)    
    x = preprocessed.values
    docs = dict(zip(list(range(len(x))), x))
    bert = BERTTransformer(docs, model) #"xlm-roberta-base")
    bert.fit()
    x = bert.to_matrix()
    return x

def train(train_data = parse_data.get_train(), dev_data = parse_data.get_dev(), model = "distilbert-base-nli-mean-tokens" ):
    #Prepare the train data
    train_texts = train_data["text_a"].to_list()
    train_y = train_data['label'].to_list()
    train_matrix = fit_space(train_texts, model) 
    del train_texts       
       
    parameters = {"C":[0.1,1,10,25,50,100,500],"penalty":["l1","l2"]}
    lr_learner = LogisticRegression(max_iter = 100000,  solver="saga")
    gs = GridSearchCV(lr_learner, parameters, verbose = 10, n_jobs = 8,cv = 10, refit = True)
    gs.fit(train_matrix, train_y)
    clf = gs.best_estimator_
    scores = cross_val_score(clf, train_matrix, train_y, cv=10, scoring='f1_macro')
    logging.info("TRAIN SGD 10fCV F1-score: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    clf = clf.fit(train_matrix, train_y)
    # Prepare output
    #fitted = clf.fit(train_matrix, train_y)
    with open(os.path.join(config.PICKLES_PATH, model + "_clf.pkl"), "wb") as f:
        pickle.dump(clf, f)
    return
    predictions = clf.predict(train_matrix)
    acc_svm = f1_score(train_y, predictions)
    logging.info("TRAIN SGD dataset prediction: %0.4f" % acc_svm)
    #Train the dev data
    dev_texts = dev_data["text_a"].to_list()
    dev_y = dev_data['label'].to_list()
    dev_matrix = fit_space(dev_texts) 
    del dev_texts
    
    #Evaluate the dev data
    predictions = clf.predict(dev_matrix)
    acc_svm = f1_score(dev_y, predictions)
    logging.info("DEV SGD dataset prediction: %0.4f" % acc_svm)

 
    return clf

def evaluate(test_data=parse_data.get_test(), mname = "distilbert-base-nli-mean-tokens" ):
    with open(os.path.join(config.PICKLES_PATH, mname + "_clf.pkl"), "rb") as f:
        model = pickle.load(f)
    _fit(model, test_data, mname)

def _fit(model, test_data=parse_data.get_test(), mname = "distilbert-base-nli-mean-tokens"):
    X = fit_space(test_data["text_a"].to_list(), mname)
    orig = test_data['label'].to_list()
    preds = model.predict(X)
    print(f1_score(orig,preds))
    return 

def fit(texts, mname="xlm-r-large-en-ko-nli-ststb"):
    X = fit_space(texts, mname)
    with open(os.path.join(config.PICKLES_PATH, mname + "_clf.pkl"), "rb") as f:
       model = pickle.load(f)    
    predictions = model.predict(X)    
    return predictions

def fit_probs(texts, mname="xlm-r-large-en-ko-nli-ststb"):
    features = fit_space(texts, mname)
    with open(os.path.join(config.PICKLES_PATH, mname + "_clf.pkl"), "rb") as f:
       model = pickle.load(f)    
    try:
        predictions = model.decision_function(features)
    except:
        predictions = model.predict_proba(features)   
    return predictions
"""
#()
#train(model="roberta-large-nli-stsb-mean-tokens")
print("TRAIN: ")
evaluate(parse_data.get_train(), mname="xlm-r-large-en-ko-nli-ststb")
print("DEV: ")
evaluate(parse_data.get_dev(), mname="xlm-r-large-en-ko-nli-ststb")
print("TEST: ")
evaluate(parse_data.get_test(), mname="xlm-r-large-en-ko-nli-ststb")
"""
