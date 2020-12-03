# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:13:53 2020

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
    
def prepare_all():
    train = parse_data.get_train()
    dev = parse_data.get_dev()
    test = parse_data.get_test()    
    X = train["text_a"].to_list() + dev["text_a"].to_list() + test["text_a"].to_list() 
    y = train["label"].to_list() + dev["label"].to_list() + test["label"].to_list()     
    return X,y

def prepare_text(texts, model = "distilbert-base-nli-mean-tokens"):
    preprocessed = preprocess_texts(texts)    
    x = preprocessed.values
    docs = dict(zip(list(range(len(x))), x))
    bert = BERTTransformer(docs, model) #"xlm-roberta-base")
    bert.fit()
    x = bert.to_matrix()
    return x

def train(X, final_y, model = "distilbert-base-nli-mean-tokens" ):
    #Prepare the train data
    train_matrix = prepare_text(X, model) 
       
    parameters = {"loss":["hinge","log"],"penalty":["elasticnet"],"alpha":[0.01,0.001,0.0001,0.0005],"l1_ratio":[0.05,0.25,0.3,0.6,0.8,0.95],"power_t":[0.5,0.1,0.9]}
    svc = SGDClassifier()
    gs1 = GridSearchCV(svc, parameters, verbose = 1, n_jobs = 8,cv = 10, refit = True)     
    gs1.fit(train_matrix, final_y)
    clf = gs1.best_estimator_    
    scores = cross_val_score(clf, train_matrix, final_y, n_jobs = 8, verbose = 1, cv=10, scoring='f1_macro')
    acc_svm = scores.mean()
    logging.info("TRAIN SGD 10fCV F1-score: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                 
    parameters = {"C":[0.1,1,10,25,50,100,500],"penalty":["l1","l2"]}
    """
    lr_learner = LogisticRegression(max_iter = 100000,  solver="saga")
    gs = GridSearchCV(lr_learner, parameters, verbose = 1, n_jobs = 8,cv = 10, refit = True)
    gs.fit(train_matrix, final_y)
    clf = gs.best_estimator_
    acc_lr = cross_val_score(clf, train_matrix, y, n_jobs = 8, verbose = 1, cv=10, scoring='f1_macro')
    logging.info("TRAIN SGD 10fCV F1-score: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    
    if acc_svm > acc_lr:
        clf = clf1
    """
    
    # Prepare output
    #fitted = clf.fit(train_matrix, train_y)
    with open(os.path.join(config.PICKLES_PATH, model + "_cv_clf.pkl"), "wb") as f:
        pickle.dump(clf, f)
    return
    return clf

def evaluate(test_data=parse_data.get_test(), mname = ""):
    with open(os.path.join(config.PICKLES_PATH, mname + "_cv_clf.pkl"), "rb") as f:
        model = pickle.load(f)
    fit(model, test_data, mname)
    
def fit(model, test_data=parse_data.get_test(), mname = "distilbert-base-nli-mean-tokens"):
    X = prepare_text(test_data["text_a"].to_list(), mname)
    orig = test_data['label'].to_list()
    preds = model.predict(X)
    print(f1_score(orig,preds))

X, y = prepare_all()
for model in ["roberta-large-nli-stsb-mean-tokens", "xlm-r-large-en-ko-nli-ststb", "distilbert-base-nli-mean-tokens"]:
    train(X, y, model=model)

