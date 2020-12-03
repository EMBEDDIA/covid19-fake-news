
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

def load(path = "pickles"):
    with open(os.path.join(path,"tax2vec.pkl"), "rb") as f:
        tax2vec = pickle.load(f)
    with open(os.path.join(path,"tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    return tax2vec, tfidf

def save(tax2vec, tfidf, path = "pickles"):
    with open(os.path.join(path,"tax2vec.pkl"), "wb") as f:
        pickle.dump(tax2vec, f)
    with open(os.path.join(path,"tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)


def fit_space(train_texts, path = "pickles", mode="ALL"):
    tax2vec_instance, tokenizer = load(path)
    semantic_features = tax2vec_instance.transform(train_texts)
    densify = np.array(semantic_features.todense())
    return densify

    
def train(train_data = parse_data.get_train(), dev_data = parse_data.get_dev()):
    train_texts = train_data["text_a"].to_list()
    train_y = train_data['label'].to_list()
    
    dev_texts = dev_data["text_a"].to_list()
    dev_y = dev_data['label'].to_list()
    
    tax2vec_dict = dict()
    for _ in range(1):
        #for num_features in [10,25,50,75,100]:
        for num_features in [ 10 ]:
            tax2vec_instance = tax2vec.tax2vec(
                max_features=num_features,
                targets=train_y,
                num_cpu="all",
                heuristic="closeness_centrality",
                class_names=list(set(train_y))
                )            
            
            tax2vec_instance.fit(train_texts)
            _, tokenizer, _ = data_docs_to_matrix(train_texts, mode="matrix_pan")

            save(tax2vec_instance, tokenizer)
            
            return
            features_train = fit_space(train_texts)
            features_dev = fit_space(dev_texts)
 
            model = learn(features_train, train_y)
            score = evaluate(model,features_dev, dev_y)
            tax2vec_dict[score] = (tax2vec_instance, tokenizer, model)
        
    best_ = max(list(tax2vec_dict.keys()))
    tax2vec_instance, tokenizer, model = tax2vec_dict[best_]
    save(tax2vec, tokenizer, model)
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
    parameters = {"C":[0.1,1],"penalty":["l1"]}
    lr_learner = LogisticRegression(max_iter = 100000,  solver="saga")
    #gs = GridSearchCV(lr_learner, parameters, verbose = 0, n_jobs = 8,cv = 10, refit = True)
    bs = GridSearchCV(estimator=lr_learner, search_spaces=parameters, n_jobs=8, cv=10)
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

"""
if __name__ == "__main__":
    data_test = parse_data.get_test()
    data_validation = parse_data.get_dev()
    data_train = parse_data.get_train()
    
    train_text = ["Brexit (/ˈbrɛksɪt, ˈbrɛɡzɪt/;[1] a portmanteau of British and exit) is the withdrawal of the United Kingdom (UK) from the European Union (EU). Following a referendum held on 23 June 2016 in which 51.9 per cent of those voting supported leaving the EU, the Government invoked Article 50 of the Treaty on European Union, starting a two-year process which was due to conclude with the UK's exit on 29 March 2019 – a deadline which has since been extended to 31 October 2019.[2]","Withdrawal from the EU has been advocated by both left-wing and right-wing Eurosceptics, while pro-Europeanists, who also span the political spectrum, have advocated continued membership and maintaining the customs union and single market. The UK joined the European Communities (EC) in 1973 under the Conservative government of Edward Heath, with continued membership endorsed by a referendum in 1975. In the 1970s and 1980s, withdrawal from the EC was advocated mainly by the political left, with the Labour Party's 1983 election manifesto advocating full withdrawal. From the 1990s, opposition to further European integration came mainly from the right, and divisions within the Conservative Party led to rebellion over the Maastricht Treaty in 1992. The growth of the UK Independence Party (UKIP) in the early 2010s and the influence of the cross-party People's Pledge campaign have been described as influential in bringing about a referendum. The Conservative Prime Minister, David Cameron, pledged during the campaign for the 2015 general election to hold a new referendum—a promise which he fulfilled in 2016 following pressure from the Eurosceptic wing of his party. Cameron, who had campaigned to remain, resigned after the result and was succeeded by Theresa May, his former Home Secretary. She called a snap general election less than a year later but lost her overall majority. Her minority government is supported in key votes by the Democratic Unionist Party.","The broad consensus among economists is that Brexit will likely reduce the UK's real per capita income in the medium term and long term, and that the Brexit referendum itself damaged the economy.[a] Studies on effects since the referendum show a reduction in GDP, trade and investment, as well as household losses from increased inflation. Brexit is likely to reduce immigration from European Economic Area (EEA) countries to the UK, and poses challenges for UK higher education and academic research. As of May 2019, the size of the divorce bill—the UK's inheritance of existing EU trade agreements—and relations with Ireland and other EU member states remains uncertain. The precise impact on the UK depends on whether the process will be a hard or soft Brexit."]

    train()
    print(fit_space(train_text, mode="T2V"))
 """