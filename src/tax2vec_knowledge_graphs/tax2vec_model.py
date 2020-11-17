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


def get_features(data_train, data_validation):
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
        print(semantic_features_train)

        ## get test features
        train_matrices_for_svm = []
        test_matrices_for_svm = []
        semantic_features_test = tax2vec_instance.transform(data_validation['text_a'])

        train_matrices_for_svm.append(semantic_features_train)
        test_matrices_for_svm.append(semantic_features_test)

        tfidf_word_train, tokenizer_2, _ = data_docs_to_matrix(
            data_train['text_a'], mode="matrix_pan")
        tfidf_word_test = tokenizer_2.transform(build_dataframe(data_validation['text_a']))
        train_matrices_for_svm.append(tfidf_word_train)
        test_matrices_for_svm.append(tfidf_word_test)

        ## stack features (sparse)
        features_train = hstack(train_matrices_for_svm)
        features_test = hstack(test_matrices_for_svm)

    return features_train, features_test

def fit(X, y_train, X_validation, Y_validation):
    new_train_y = []

    for y in y_train:
        if isinstance(y, list):
            new_train_y.append(list(y).index(1))
        else:
            new_train_y.append(y)

    classifiers = [GradientBoostingClassifier(), RandomForestClassifier(n_estimators=10), svm.SVC(probability=True), SGDClassifier()]  ## spyct.Model()
    best_classifier = classifiers[0]
    best_score = 0
    for classifier in range(len(classifiers)):
        parameters = {}
        if classifier == 0:
            parameters = {"loss":["deviance", "exponential"],"learning_rate":[0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],"n_estimators":[10, 20, 50, 100, 200]}
        elif classifier == 1:
            parameters = {"n_estimators":[10, 20, 50, 100, 200]}
        elif classifier == 2:
            parameters = {"C": [0.1, 1, 10, 25, 50, 100, 500], "kernel": ["linear", "poly", "rbf", "sigmoid"]}
        elif classifier == 3:
            parameters = {"loss": ["hinge", "log", "huber"], "penalty": ["l2", "l1", "elasticnet"]}

        clf = BayesSearchCV(estimator=classifiers[classifier], search_spaces=parameters, n_jobs=-8, cv=10)
        clf.fit(X, new_train_y)

        clf_score = evaluate(clf, X_validation, Y_validation)
        if clf_score > best_score:
            best_score = clf_score
            best_classifier = clf
    return best_classifier

def evaluate(clf, X, test_y):
    print(len(test_y))
    new_test_y = []
    for y in test_y:
        if isinstance(y, list):
            new_test_y.append(list(y).index(1))
        else:
            new_test_y.append(y)

    print(len(new_test_y))
    y_pred = clf.predict(X)
    print(len(y_pred))
    copt = f1_score(new_test_y, y_pred, average='micro')
    #print("Current score {}".format(copt))
    return copt


if __name__ == "__main__":
    data_test = parse_data.get_test()
    data_validation = parse_data.get_dev()
    data_train = parse_data.get_train()

    #_, features_validation = get_features(data_train, data_validation)
    #features_train, features_test = get_features(data_train, data_test)

    #pd.DataFrame(features_train.toarray()).to_csv("train_features_kg.csv")
    #pd.DataFrame(features_validation.toarray()).to_csv("validation_features_kg.csv")
    #pd.DataFrame(features_test.toarray()).to_csv("test_features_kg.csv")
    
    #features_train = []
    #features_test = []
    
    s = pd.read_csv('features/train_features_kg.csv', sep=',')
    features_train = pd.DataFrame(s.to_numpy())

    s = pd.read_csv('features/validation_features_kg.csv', sep=',')
    features_validation = pd.DataFrame(s.to_numpy())

    s = pd.read_csv('features/test_features_kg.csv', sep=',')
    features_test = pd.DataFrame(s.to_numpy())

    print(len(data_validation['label']))
    print(features_train.size)
    print(features_train.shape)



    model = fit(features_train, data_train['label'].to_list(), features_validation, data_validation['label'].to_list())
    evaluate(model, features_test, data_test['label'].to_list())

    ## save model with pickle
    with open(os.path.join("clf_en.pkl"), mode='wb') as f:
        pickle.dump(model, f)
