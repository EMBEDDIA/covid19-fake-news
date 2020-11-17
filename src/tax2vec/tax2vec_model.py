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
        print(semantic_features_train)

        ## get test features
        train_matrices_for_svm = []
        validation_matrices_for_svm = []
        test_matrices_for_svm = []
        semantic_features_validation = tax2vec_instance.transform(data_validation['text_a'])
        semantic_features_test = tax2vec_instance.transform(data_test['text_a'])

        train_matrices_for_svm.append(semantic_features_train)
        validation_matrices_for_svm.append(semantic_features_validation)
        test_matrices_for_svm.append(semantic_features_test)

        tfidf_word_train, tokenizer_2, _ = data_docs_to_matrix(
            data_train['text_a'], mode="matrix_pan")
        tfidf_word_validation = tokenizer_2.transform(build_dataframe(data_validation['text_a']))
        tfidf_word_test = tokenizer_2.transform(build_dataframe(data_test['text_a']))
        train_matrices_for_svm.append(tfidf_word_train)
        validation_matrices_for_svm.append(tfidf_word_validation)
        test_matrices_for_svm.append(tfidf_word_test)

        ## stack features (sparse)
        features_train = hstack(train_matrices_for_svm)
        features_validation = hstack(validation_matrices_for_svm)
        features_test = hstack(test_matrices_for_svm)

    return features_train,features_validation, features_test
    
def train(train_data = parse_data.get_train(), dev_data = parse_data.get_dev()):
    train_texts = train_data["text_a"].to_list()
    train_y = train_data['label'].to_list()
    
    dev_texts = dev_data["text_a"].to_list()
    dev_y = dev_data['label'].to_list()
    
    tax2vec_dict = dict()
    for _ in range(1):
        #for num_features in [10,25,50,75,100]:
        for num_features in [ 25 ]:
            tax2vec_instance = tax2vec.tax2vec(
                max_features=num_features,
                targets=train_y,
                num_cpu="all",
                heuristic="closeness_centrality",
                class_names=list(set(train_y))
                )            
            
            tax2vec_instance.fit(train_texts)
            _, tokenizer, _ = data_docs_to_matrix(train_texts, mode="matrix_pan")

            features_train = get_features(train_texts, tokenizer, tax2vec_instance)
            features_dev = get_features(dev_texts, tokenizer, tax2vec_instance)
            
            print(features_train)
            with open("features_train","wb") as f:
                pickle.dump(features_train,f)

            with open("features_dev","wb") as f:
                pickle.dump(features_dev,f)
            return
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
    for classifier in classifiers:
        clf = classifier
        clf.fit(X, new_train_y)

        clf_score = evaluate(clf, X_validation, Y_validation)
        if clf_score > best_score:
            best_score = clf_score
            best_classifier = clf
    return best_classifier

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
    features_train = pd.DataFrame(s.to_numpy())

    s = pd.read_csv('features/validation_features.csv', sep=',')
    features_validation = pd.DataFrame(s.to_numpy())

    s = pd.read_csv('features/test_features.csv', sep=',')
    features_test = pd.DataFrame(s.to_numpy())

    model = pickle.load(open(os.path.join(config.PICKLES_PATH, "clf_" + "en" + ".pkl"), 'rb'))

    print(features_train.shape)
    print(len(data_train['label']))
    print(features_validation.shape)
    print(len(data_validation['label']))
    print(features_test.shape)
    print(len(data_test['label']))


    #model = fit(features_train, data_train['label'].to_list(), features_validation, data_validation['label'].to_list())
    print("Evaluating test set:")
    evaluate(model, features_test, data_test['label'].to_list())

    ## save model with pickle
    #with open(os.path.join("clf_en.pkl"), mode='wb') as f:
     #   pickle.dump(model, f)