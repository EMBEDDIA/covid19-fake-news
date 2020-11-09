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
import tax2vec as t2v
from preprocessing import *
from tax2vec.preprocessing import *
from tax2vec.models import *


def get_features(data_train, data_validation):
    ## generate learning examples
    num_splits = 1

    ## do the stratified shufflesplit
    for _ in range(num_splits):
        tax2vec_instance = t2v.tax2vec(
            max_features=50,
            targets=data_train['label'].to_list(),
            num_cpu=2,
            heuristic="closeness_centrality",
            class_names=list(set(data_train['label'].to_list())))
        semantic_features_train = tax2vec_instance.fit_transform(data_train['tweet'])
        print(semantic_features_train)

        ## get test features
        train_matrices_for_svm = []
        test_matrices_for_svm = []
        semantic_features_test = tax2vec_instance.transform(data_validation['tweet'])

        train_matrices_for_svm.append(semantic_features_train)
        test_matrices_for_svm.append(semantic_features_test)

        tfidf_word_train, tokenizer_2, _ = data_docs_to_matrix(
            data_train['tweet'], mode="matrix_pan")
        tfidf_word_test = tokenizer_2.transform(build_dataframe(data_validation['tweet']))
        train_matrices_for_svm.append(tfidf_word_train)
        test_matrices_for_svm.append(tfidf_word_test)

        ## stack features (sparse)
        features_train = hstack(train_matrices_for_svm)
        features_test = hstack(test_matrices_for_svm)

    return features_train, features_test


def fit(X, y_train, cparam=50, max_iter = 5000):
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


if __name__ == "__main__":
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
