## tax2vec

import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
import parse_data
import time
import csv
import config
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
import pickle
from preprocessing import *
import tax2vec
from tax2vec.preprocessing import *
from tax2vec.models import *


def get_features(text_list, tokenizer, tax2vec_instance):
    features_all = []
    semantic_features_test = tax2vec_instance.transform(text_list)
    features_all.append(semantic_features_test)
    tfidf_features = tokenizer.transform(build_dataframe(text_list))
    features_all.append(tfidf_features)
    features = np.hstack(features_all)    
    return features
    
def train(train_data = parse_data.get_train(), dev_data = parse_data.get_dev()):
    train_texts = train_data["text_a"].to_list()
    train_y = train_data['label'].to_list()
    
    dev_texts = dev_data["text_a"].to_list()
    dev_y = dev_data['label'].to_list()
    
    tax2vec_dict = dict()
    for _ in range(1):
        for num_features in [10,25,50,75,100]:
            tax2vec_instance = tax2vec.tax2vec(
                max_features=num_features,
                targets=train_y,
                num_cpu=8,
                heuristic="closeness_centrality",
                class_names=list(set(train_y))
                )            
            
            tax2vec_instance.fit(train_texts)
            _, tokenizer, _ = data_docs_to_matrix(train_texts, mode="matrix_pan")

            features_train = get_features(train_texts, tokenizer, tax2vec_instance)
            features_dev = get_features(dev_texts, tokenizer, tax2vec_instance)
            
            with open("features_train","wb") as f:
                pickle.dump(features_train,f)

            with open("features_dev","wb") as f:
                pickle.dump(features_dev,f)

            model = learn(features_train, train_y)
            score = evaluate(model,features_dev, dev_y)
            tax2vec_dict[score] = (tax2vec_instance, tokenizer, model)
        
    best_ = max(list(tax2vec_dict.keys()))
    tax2vec_instance, tokenizer, model = tax2vec_dict[best_]
    
    return tax2vec_instance, tokenizer, model
    
def export():
    tax2vec_instance, tokenizer, model = train(parse_data.get_train(), parse_data.get_dev())
    with open(os.path.join(config.PICKLES_PATH, "tax2vec_instance.pkl"),mode='wb') as f:
        pickle.dump(tax2vec_instance,f)
    with open(os.path.join(config.PICKLES_PATH, "tokenizer.pkl"),mode='wb') as f:
        pickle.dump(tokenizer,f)
    with open(os.path.join(config.PICKLES_PATH, "model.pkl"),mode='wb') as f:
        pickle.dump(model,f)


def learn(train_matrix, train_y):
    parameters = {"C":[0.1,1,10,25,50,100,500],"penalty":["l1","l2"]}
    lr_learner = LogisticRegression(max_iter = 100000,  solver="saga")
    gs = GridSearchCV(lr_learner, parameters, verbose = 0, n_jobs = 8,cv = 10, refit = True)
    gs.fit(train_matrix, train_y)
    clf = gs.best_estimator_
    scores = cross_val_score(clf, train_matrix, train_y, cv=10, scoring='f1_macro')
    print("TRAIN SGD 10fCV F1-score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return clf.fit(train_matrix, train_y)

    
def fit(X):
    
    new_train_y = []

    for y in y_train:
        if isinstance(y, list):
            new_train_y.append(list(y).index(1))
        else:
            new_train_y.append(y)
    clf = svm.LinearSVC(C=cparam, max_iter = max_iter)
    clf.fit(X, new_train_y)
    return clf

def evaluate(clf, X, test_y):
    new_test_y = []
    for y in test_y:
        if isinstance(y, list):
            new_test_y.append(list(y).index(1))
        else:
            new_test_y.append(y)

    y_pred = clf.predict(X)
    copt = f1_score(new_test_y, y_pred, average='micro')
    print("Current score {}".format(copt))
    return copt

if __name__ == "__main__":
    
    tax2vec_instance, tokenizer, model = train()
    """
    data_validation = parse_data.readValidation()
    data_train = parse_data.readTrain()

    
    features_train, features_test = get_features(data_train, data_validation)

    pd.DataFrame(features_train.toarray()).to_csv("train_features.csv")
    pd.DataFrame(features_test.toarray()).to_csv("test_features.csv")


    model = fit(features_train, data_train['label'].to_list())
    evaluate(model, features_test, data_validation['label'].to_list())

    ## save model with pickle
    with open(os.path.join("clf_en.pkl"), mode='wb') as f:
        pickle.dump(model, f)
    """