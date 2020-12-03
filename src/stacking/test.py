# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:02:22 2020

@author: Bosec
"""
import os
import numpy as np
import parse_data
import pandas as pd
import lsa_model
import statistical_features
from model import train_NN, predict
from model_helper import ColumnarDataset
from sklearn import preprocessing
import sentence_transfomers
import statistical_baseline
import torch 
import distilBERT_model as db
import tax2vec_model as t2v_vanila
import tax2vec_model_kg as t2v_kg
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

text = "ncludes high-quality download in MP3, FLAC and more. Paying supporters also get unlimited streaming via the free Bandcamp app."


def get_features(texts):
    lsa_features = lsa_model.fit(texts).reshape((len(texts), 1))
    #dbert_features = db.fit(texts)
    vanila = t2v_vanila.fit_probs(texts)
    #kg =  t2v_kg.fit_probs(texts)
    bert_features = sentence_transfomers.fit(texts, "distilbert-base-nli-mean-tokens").reshape((len(texts), 1))
    stat_features = statistical_baseline.fit(texts).reshape((len(texts), 1)).reshape((len(texts), 1))
    roberta_features = sentence_transfomers.fit(texts, "roberta-large-nli-stsb-mean-tokens").reshape((len(texts), 1))
    xlm_features = sentence_transfomers.fit(texts, "xlm-r-large-en-ko-nli-ststb").reshape((len(texts), 1))    
    features = np.hstack(([lsa_features, vanila, bert_features, stat_features, roberta_features, xlm_features, roberta_features]))
    print(features.shape)
    return preprocessing.scale(features)
    
def get_features_probs(texts):
    lsa_features = lsa_model.fit_probs(texts).reshape((len(texts), 1))
    #dbert_features = db.fit(texts)
    vanila = t2v_vanila.fit_probs(texts)
    #kg =  t2v_kg.fit_probs(texts)
    bert_features = sentence_transfomers.fit_probs(texts, "distilbert-base-nli-mean-tokens").reshape((len(texts), 1))
    stat_features = statistical_baseline.fit_probs(texts).reshape((len(texts), 1))
    roberta_features = sentence_transfomers.fit_probs(texts, "roberta-large-nli-stsb-mean-tokens").reshape((len(texts), 1))
    xlm_features = sentence_transfomers.fit_probs(texts, "xlm-r-large-en-ko-nli-ststb").reshape((len(texts), 1))    
    features = np.hstack(([lsa_features, vanila, bert_features, stat_features, roberta_features, xlm_features, roberta_features]))
    print(features.shape)
    return preprocessing.scale(features)

def prepare_dataset(data, prep = get_features_probs):
    train_texts = data["text_a"].to_list()
    train_y = data['label'].to_list()
    train_matrix = prep(train_texts)     
    train_matrix = pd.DataFrame(train_matrix)
    print(train_matrix)
    train_dataset = ColumnarDataset(train_matrix, train_y)
    return train_dataset, train_matrix.shape[1]      

def train_nets(train_data=parse_data.get_train(), dev_data = parse_data.get_dev(), test_data = parse_data.get_test()):
    train_dataset, dims = prepare_dataset(train_data, prep = get_features)
    valid_dataset, dims = prepare_dataset(dev_data, prep = get_features)   
    test_dataset, dims = prepare_dataset(test_data, prep = get_features)
    best  = 0
    outs = None
    best_lr = 0
    best_p = 0
    for lr in [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        for p in [0.1, 0.3, 0.5, 0.7]:
            print(lr, p)
            net = train_NN(train_dataset, valid_dataset, test_dataset, dims ,max_epochs=20, batch_size = 64, lr = lr, dropout = p)
            score = predict(test_dataset, net)
            if score > best:
                best = score
                outs = net
                best_lr = lr
                best_p = p
    torch.save(outs.state_dict(), os.path.join("pickles","3net"+str(best_lr)+"_"+str(best_p)+".pymodel"))

def train_SGD():
    train = parse_data.get_train()
    train_texts = train["text_a"].to_list()
    train_features = get_features(train_texts)
    train_y = train["label"].to_list()
    del train_texts
    del train
    print("TRAINING: ")
    parameters = {"loss":["hinge","log"],"penalty":["elasticnet"],"alpha":[0.01,0.001,0.0001,0.0005],"l1_ratio":[0.05,0.25,0.3,0.6,0.8,0.95],"power_t":[0.5,0.1,0.9]}
    x = SGDClassifier()
    clf1 = GridSearchCV(x, parameters, n_jobs = 8,cv = 10, refit = True, verbose = 10)
    scores = cross_val_score(clf1, train_features, train_y, n_jobs = 8, cv=10, scoring='f1_macro', verbose = 10)
    clf1 = clf1.fit(train_features, train_y)
    import pickle
    with open("clf_all_2.pkl","wb") as f:
        pickle.dump(clf1, f)
        
    score_train = f1_score(train_y, clf1.predict(train_features))
    
    dev = parse_data.get_dev()
    dev_texts = dev["text_a"].to_list()
    dev_features = get_features(dev_texts)
    dev_y = dev["label"].to_list()
    del dev_texts
    del dev

    dev_score = f1_score(dev_y, clf1.predict(dev_features))

    test = parse_data.get_test()
    test_texts = test["text_a"].to_list()
    test_features = get_features(test_texts)
    test_y = test["label"].to_list()
    del test_texts
    del test

    test_score = f1_score(test_y, clf1.predict(test_features))

    print("TRAIN: %0.4f DEV: %0.4f TEST: %0.4f" % (score_train, dev_score, test_score))
import pickle
def predict_valid():
    vals = parse_data.readValidation()
    feats = get_features(vals["tweet"])
    with open("clf_all.pkl","rb") as f:
        clf = pickle.load(f)
    predictions = clf.predict(feats)
    ou = ["fake" if p == 0 else "real" for p in predictions ]
    outs = {"id": list(range(1,len(ou)+1)), "label" : ou}
    df = pd.DataFrame(outs)
    df.to_csv("answer.txt", index=False)
        
#predict_valid()    
train_SGD()
#train_nets()

