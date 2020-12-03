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


def load(path="pickles", lang='en',path_in="."):
    with open(os.path.join(path, "tax2vec_kg.pkl"), "rb") as f:
        tax2vec = pickle.load(f)
    tokenizer = pickle.load(open(os.path.join(config.PICKLES_PATH, "tokenizer" + ".pkl"), 'rb'))
    return tax2vec, tokenizer


def save(tax2vec, tokenizer, path="pickles"):
    with open(os.path.join(path, "tax2vec_kg.pkl"), "wb") as f:
        pickle.dump(tax2vec, f)
    with open(os.path.join(path, "tokenizer_kg.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

def _import():
    model = pickle.load(open(os.path.join(config.PICKLES_PATH, "clf_en_kg.pkl"), 'rb'))
    return model

def fit_space(X, model_path="."):
    df_final = build_dataframe(X)
    t2v_instance, tokenizer = load()
    features_matrix = []
    semantic_features = t2v_instance.transform(df_final)
    features_matrix.append(semantic_features)
    tfidf_words = tokenizer.transform(df_final)
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
            max_features=10,
            num_cpu=2,
            heuristic="closeness_centrality",
            disambiguation_window=2,
            start_term_depth=3,
            mode="index_word",
            simple_clean=True,
            knowledge_graph=True,
            hyp=100,
            path='data-concept/refined.txt'
        )
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

    return features_train, features_validation, features_test

def train(X, y_train, X_validation, Y_validation):
    new_train_y = []

    for y in y_train:
        if isinstance(y, list):
            new_train_y.append(list(y).index(1))
        else:
            new_train_y.append(y)

    classifiers = [GradientBoostingClassifier(), RandomForestClassifier(n_estimators=10), svm.SVC(probability=True), SGDClassifier()]
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

    #pd.DataFrame(features_train.toarray()).to_csv("train_features_kg.csv")
    #pd.DataFrame(features_validation.toarray()).to_csv("validation_features_kg.csv")
    #pd.DataFrame(features_test.toarray()).to_csv("test_features_kg.csv")

    #features_train = []
    #features_test = []


    #s = pd.read_csv('features/train_features_kg.csv', sep=',')
    #features_train = pd.DataFrame(s.to_numpy())

    #s = pd.read_csv('features/validation_features_kg.csv', sep=',')
    #features_validation = pd.DataFrame(s.to_numpy())

    #s = pd.read_csv('features/test_features_kg.csv', sep=',')
    #features_test = pd.DataFrame(s.to_numpy())



    #model = train(features_train, data_train['label'].to_list(), features_validation, data_validation['label'].to_list())
    #print("Evaluating on test set...")
    #evaluate(model, features_test, data_test['label'].to_list())

    ## save model with pickle
    #with open(os.path.join("clf_en.pkl"), mode='wb') as f:
    #    pickle.dump(model, f)

    train_text = [
        "Brexit (/ˈbrɛksɪt, ˈbrɛɡzɪt/;[1] a portmanteau of British and exit) is the withdrawal of the United Kingdom (UK) from the European Union (EU). Following a referendum held on 23 June 2016 in which 51.9 per cent of those voting supported leaving the EU, the Government invoked Article 50 of the Treaty on European Union, starting a two-year process which was due to conclude with the UK's exit on 29 March 2019 – a deadline which has since been extended to 31 October 2019.[2]",
        "Withdrawal from the EU has been advocated by both left-wing and right-wing Eurosceptics, while pro-Europeanists, who also span the political spectrum, have advocated continued membership and maintaining the customs union and single market. The UK joined the European Communities (EC) in 1973 under the Conservative government of Edward Heath, with continued membership endorsed by a referendum in 1975. In the 1970s and 1980s, withdrawal from the EC was advocated mainly by the political left, with the Labour Party's 1983 election manifesto advocating full withdrawal. From the 1990s, opposition to further European integration came mainly from the right, and divisions within the Conservative Party led to rebellion over the Maastricht Treaty in 1992. The growth of the UK Independence Party (UKIP) in the early 2010s and the influence of the cross-party People's Pledge campaign have been described as influential in bringing about a referendum. The Conservative Prime Minister, David Cameron, pledged during the campaign for the 2015 general election to hold a new referendum—a promise which he fulfilled in 2016 following pressure from the Eurosceptic wing of his party. Cameron, who had campaigned to remain, resigned after the result and was succeeded by Theresa May, his former Home Secretary. She called a snap general election less than a year later but lost her overall majority. Her minority government is supported in key votes by the Democratic Unionist Party.",
        "The broad consensus among economists is that Brexit will likely reduce the UK's real per capita income in the medium term and long term, and that the Brexit referendum itself damaged the economy.[a] Studies on effects since the referendum show a reduction in GDP, trade and investment, as well as household losses from increased inflation. Brexit is likely to reduce immigration from European Economic Area (EEA) countries to the UK, and poses challenges for UK higher education and academic research. As of May 2019, the size of the divorce bill—the UK's inheritance of existing EU trade agreements—and relations with Ireland and other EU member states remains uncertain. The precise impact on the UK depends on whether the process will be a hard or soft Brexit."]

    print(fit(train_text))
