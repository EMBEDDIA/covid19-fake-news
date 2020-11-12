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
    
def prepare_text(texts, model = "distilbert-base-nli-mean-tokens"):
    preprocessed = preprocess_texts(texts)    
    x = preprocessed.values
    docs = dict(zip(list(range(len(x))), x))
    bert = BERTTransformer(docs, model) #"xlm-roberta-base")
    bert.fit()
    x = bert.to_matrix()
    return x

def train(train_data = parse_data.get_train(), dev_data = parse_data.get_dev()):
    #Prepare the train data
    train_texts = train_data["text_a"].to_list()
    train_y = train_data['label'].to_list()
    train_matrix = prepare_text(train_texts) 
    del train_texts       
       
    parameters = {"C":[0.1,1,10,25,50,100,500],"penalty":["l1","l2"]}
    lr_learner = LogisticRegression(max_iter = 100000,  solver="saga")
    gs = GridSearchCV(lr_learner, parameters, verbose = 10, n_jobs = 8,cv = 10, refit = True)
    gs.fit(train_matrix, train_y)
    clf = gs.best_estimator_
    scores = cross_val_score(clf, train_matrix, train_y, cv=10, scoring='f1_macro')
    logging.info("TRAIN SGD 10fCV F1-score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    #Train the dev data
    dev_texts = dev_data["text_a"].to_list()
    dev_y = dev_data['label'].to_list()
    dev_matrix = prepare_text(dev_texts) 
    del dev_texts
    
    #Evaluate the dev data
    predictions = clf.predict(dev_matrix)
    acc_svm = f1_score(predictions, dev_y)
    logging.info("DEV SGD dataset prediction: %0.2f" % acc_svm)

    # Prepare output
    fitted = clf.fit(train_matrix, train_y)
    with open(os.path.join(config.PICKLES_PATH, "clf.pkl"), "wb") as f:
        pickle.dump(fitted, f)
    return fitted


def fit(model=train(), test_data=parse_data.get_test()):
    X = test_data["text_a"].to_list()
    orig = test_data['label'].to_list()
    preds = model.predict(X)
    print(f1_score(orig,preds))

fit()