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
from scipy.sparse import coo_matrix, hstack
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

import pandas as pd
def load(path="pickles"):
    with open(os.path.join(path, "tax2vec.pkl"), "rb") as f:
        tax2vec = pickle.load(f)
    tokenizer = pickle.load(open(os.path.join(config.PICKLES_PATH, "tokenizer" + ".pkl"), 'rb'))
    return tax2vec, tokenizer


def save(tax2vec, tokenizer, path="pickles"):
    with open(os.path.join(path, "tax2vec_cv.pkl"), "wb") as f:
        pickle.dump(tax2vec, f)
    with open(os.path.join(path, "tokenizer_cv.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

def _import():
    model = pickle.load(open(os.path.join(config.PICKLES_PATH, "clf_" + "en" + ".pkl"), 'rb'))
    return model

def fit_space(X, model_path="."):
    t2v_instance, tokenizer = load()
    features_matrix = []
    semantic_features = t2v_instance.transform(X)
    features_matrix.append(semantic_features)
    tfidf_words = tokenizer.transform(build_dataframe(X))
    features_matrix.append(tfidf_words)
    features = hstack(features_matrix)
    return features

def fit_probs(texts, model_path="."):
    features = fit_space(texts, model_path)
    model = _import()
    try:
        predictions = model.decision_function(features)
    except:
        predictions = model.predict_proba(features)
    return predictions

def fit(X, model_path="."):
    reduced_matrix_form = fit_space(X, model_path)
    clf = _import()
    predictions = clf.predict(reduced_matrix_form)
    return predictions


def get_features(data_train, data_validation, data_test):
    ## generate learning examples
    num_splits = 1

    ## do the stratified shufflesplit
    for _ in range(num_splits):
        tax2vec_instance = tax2vec.tax2vec(
            max_features=50,
            targets=data_train['label'].to_list(),
            num_cpu=2,
            heuristic="closeness_centrality",
            class_names=list(set(data_train['label'].to_list())))
        semantic_features_train = tax2vec_instance.fit_transform(data_train['text_a'])

        ## get test features
        train_matrices_for_svm = []
        validation_matrices_for_svm = []
        test_matrices_for_svm = []
        semantic_features_validation = tax2vec_instance.transform(data_validation['text_a'])
        semantic_features_test = tax2vec_instance.transform(data_test['text_a'])

        train_matrices_for_svm.append(semantic_features_train)
        validation_matrices_for_svm.append(semantic_features_validation)
        test_matrices_for_svm.append(semantic_features_test)

        tfidf_word_train, tokenizer_2, _ = data_docs_to_matrix(data_train['text_a'], mode="matrix_pan")
        tfidf_word_validation = tokenizer_2.transform(build_dataframe(data_validation['text_a']))
        tfidf_word_test = tokenizer_2.transform(build_dataframe(data_test['text_a']))
        train_matrices_for_svm.append(tfidf_word_train)
        validation_matrices_for_svm.append(tfidf_word_validation)
        test_matrices_for_svm.append(tfidf_word_test)

        ## stack features (sparse)
        features_train = hstack(train_matrices_for_svm)
        features_validation = hstack(validation_matrices_for_svm)
        features_test = hstack(test_matrices_for_svm)

        save(tax2vec_instance, tokenizer_2)

    return features_train,features_validation, features_test
    
def train(X, ys):
    new_train_y = []

    for y in ys:
        if isinstance(y, list):
            new_train_y.append(list(y).index(1))
        else:
            new_train_y.append(y)

    classifiers = [GradientBoostingClassifier(), RandomForestClassifier(n_estimators=10), LogisticRegression(max_iter=1000), SGDClassifier(loss="hinge", penalty = "l2")]
    best_classifier = classifiers[0]
    best_score = 0
    for classifier in range(len(classifiers)):
        #parameters = {}
        #if classifier == 0:
        #    parameters = {"loss":["deviance", "exponential"],"learning_rate":[0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],"n_estimators":[10, 20, 50, 100, 200]}
        #elif classifier == 1:
        #    parameters = {"n_estimators":[10, 20, 50, 100, 200]}
        #elif classifier == 2:
        #    parameters = {"C": [0.1, 1, 10, 25, 50, 100, 500], "kernel": ["linear", "poly", "rbf", "sigmoid"]}
        #elif classifier == 3:
        #    parameters = {"loss": ["hinge", "log", "huber"], "penalty": ["l2", "l1", "elasticnet"]}

        #clf = BayesSearchCV(estimator=classifiers[classifier], search_spaces=parameters, n_jobs=-8, cv=10)
        #clf.fit(X, new_train_y)

        clf = classifiers[classifier]
        #clf.fit(X, new_train_y)
        clf_score = cross_val_score(clf, X, new_train_y, verbose = 1, n_jobs = -1, scoring="f1", cv = 10).mean()
        print("Scored: " + str(clf_score))
        if clf_score > best_score:
            best_score = clf_score
            best_classifier = clf

    print("Train score:")
    print(best_score)
    return_classifier = best_classifier.fit(X, new_train_y)
    return return_classifier
    
def export():
    tax2vec_instance, tokenizer, model = train(parse_data.get_train(), parse_data.get_dev())
    with open(os.path.join(config.PICKLES_PATH, "tax2vec_instance.pkl"),mode='wb') as f:
        pickle.dump(tax2vec_instance,f)
    with open(os.path.join(config.PICKLES_PATH, "tokenizer.pkl"),mode='wb') as f:
        pickle.dump(tokenizer,f)
    with open(os.path.join(config.PICKLES_PATH, "model.pkl"),mode='wb') as f:
        pickle.dump(model,f)


def learn(train_matrix, train_y):
    #train_matrix = hstack((train_matrix[0],train_matrix[1]))
    parameters = {"C":[0.1,1,10,25,50,100,500],"penalty":["l1","l2"]}
    lr_learner = LogisticRegression(max_iter = 100000,  solver="saga")
    #gs = GridSearchCV(lr_learner, parameters, verbose = 0, n_jobs = 8,cv = 10, refit = True)
    bs = BayesSearchCV(estimator=lr_learner, search_spaces=parameters, n_jobs=-8, cv=10)
    bs.fit(train_matrix, train_y)
    clf = bs.best_estimator_
    scores = cross_val_score(clf, train_matrix, train_y, cv=10, scoring='f1_macro')
    print("TRAIN SGD 10fCV F1-score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return clf.fit(train_matrix, train_y)



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
    data_test = parse_data.get_test()
    data_validation = parse_data.get_dev()
    data_train = parse_data.get_train()

    #features_train, features_validation, features_test = get_features(data_train, data_validation, data_test)
    
    #pd.DataFrame(features_train.toarray()).to_csv("train_features.csv")
    #pd.DataFrame(features_validation.toarray()).to_csv("validation_features.csv")
    #pd.DataFrame(features_test.toarray()).to_csv("test_features.csv")


    s = pd.read_csv('features/train_features.csv', sep=',')
    features_train = s.to_numpy()

    s = pd.read_csv('features/validation_features.csv', sep=',')
    features_validation = s.to_numpy()

    s = pd.read_csv('features/test_features.csv', sep=',')
    features_test = s.to_numpy()

    #model = _import()

    #model = train(features_train, data_train['label'].to_list(), features_validation, data_validation['label'].to_list())
    #print("Evaluating test set:")
    #evaluate(model, features_test, data_test['label'].to_list())

    ## save model with pickle
    features = np.vstack((features_train,features_validation))
    X = np.vstack((features, features_test))
    print(X.shape)
    print("DATA PREPARED")
    ys = data_train['label'].to_list() + data_validation['label'].to_list() + data_test['label'].to_list()
    model = train(X, ys)

    
    with open(os.path.join("clf_en_cv.pkl"), mode='wb') as f:
        pickle.dump(model, f)

    #print(fit(data_test['text_a']))