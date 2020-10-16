# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:35:34 2020

@author: Bosec
"""


## some more experiments
import xml.etree.ElementTree as ET
import config 
import numpy
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
from feature_construction import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
import pickle

try:
    import umap
except:
    pass

def train(data,output=False):
    final_y = data["label"].to_list()
    final_texts = data["tweet"].to_list()
        
    dataframe = build_dataframe(final_texts)
    print(dataframe)
    report = []

    trained_models = {}
    
    for nrep in range(1):
        for nfeat in [2500,5000,10000,15000]:
            for dim in [256,512,768]:
                tokenizer, feature_names, data_matrix = get_features(dataframe, max_num_feat = nfeat, labels = final_y)
                reducer = TruncatedSVD(n_components = min(dim, nfeat * len(feature_names)-1))
                data_matrix = reducer.fit_transform(data_matrix)
                logging.info("Generated {} features.".format(nfeat*len(feature_names)))
                parameters = {"loss":["hinge","log"],"penalty":["elasticnet"],"alpha":[0.01,0.001,0.0001,0.0005],"l1_ratio":[0.05,0.25,0.3,0.6,0.8,0.95],"power_t":[0.5,0.1,0.9]}
                svc = SGDClassifier()
                clf1 = GridSearchCV(svc, parameters, verbose = 0, n_jobs = 8,cv = 10, refit = True)
                scores = cross_val_score(clf1, data_matrix, final_y, cv=10, scoring='f1_macro')
                acc_svm = scores.mean()
                logging.info("SGD 10fCV F1-score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

                parameters = {"C":[0.1,1,10,25,50,100,500],"penalty":["l2"]}
                svc = LogisticRegression(max_iter = 100000)
                clf2 = GridSearchCV(svc, parameters, verbose = 0, n_jobs = 8,cv = 10, refit = True)
                scores = cross_val_score(clf2, data_matrix, final_y, cv=10, scoring='f1_macro')
                logging.info("LR 10fCV F1-score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                
                acc_lr = scores.mean()
                trained_models[nfeat] = ((clf1, clf2), dim)
                report.append([nfeat, acc_lr, acc_svm])
                
    
    dfx = pd.DataFrame(report)
    dfx.columns = ["Number of features","LR","SVM"]
    dfx = pd.melt(dfx, id_vars=['Number of features'], value_vars=['LR','SVM'])
    sns.lineplot(dfx['Number of features'],dfx['value'], hue = dfx["variable"], markers = True, style = dfx['variable'])
    plt.legend()
    plt.tight_layout()
    plt.savefig("tfidif-merged-cv-expanded.png",dpi = 300)
    sorted_dfx = dfx.sort_values(by = ["value"])
    print(sorted_dfx.iloc[-1,:])
    max_acc = sorted_dfx.iloc[-1,:][['Number of features','variable']]

    final_feature_number = max_acc['Number of features']
    final_learner = max_acc['variable']
    logging.info("Final feature number: {}, final learner: {}".format(final_feature_number, final_learner))
    
    if final_learner == "SVM":
        index = 0        
    else:
        index = 1

    clf_final, dim = trained_models[final_feature_number]
    clf_final = clf_final[index]
    tokenizer, feature_names, data_matrix = get_features(dataframe, max_num_feat = final_feature_number)
    reducer = TruncatedSVD(n_components = min(dim, nfeat * len(feature_names)-1)).fit(data_matrix)
    return tokenizer, clf_final, reducer

def _import(lang='en',path_in="."):
    """Imports tokenizer,clf,reducer from param(path_in, default is ../models)"""
    tokenizer = pickle.load(open("tokenizer_"+lang+".pkl",'rb'))
    clf = pickle.load(open("clf_"+lang+".pkl",'rb'))
    reducer = pickle.load(open("reducer_"+lang+".pkl",'rb'))
    return tokenizer,clf,reducer

def export():
    data = parse_data.readTrain()
    tokenizer, clf, reducer = train(data)
    with open(os.path.join("tokenizer_en.pkl"),mode='wb') as f:
        pickle.dump(tokenizer,f)
    with open(os.path.join("clf_en.pkl"),mode='wb') as f:
        pickle.dump(clf,f)
    with open(os.path.join("reducer_en.pkl"),mode='wb') as f:
        pickle.dump(reducer,f)

def fit(path=""):
    """Fits data from param(path), outputs xml file as out_path"""
    tokenizer,clf,reducer = _import()
    data = parse_data.readValidation()
    df_text = build_dataframe(data['tweet'].to_list())
    data2 = parse_data.readTrain()
    
    df_text2 = build_dataframe(data2['tweet'].to_list())

    matrix_form = tokenizer.transform(df_text2)
    reduced_matrix_form = reducer.transform(matrix_form)
    x = data2['label'].to_list() == 'real'
    x = [1 if c  == 'real' else 0 for c in data2['label'].to_list()]
    p = clf.fit(reduced_matrix_form, x)    
    matrix_form = tokenizer.transform(df_text)
    reduced_matrix_form = reducer.transform(matrix_form)
    predictions = p.predict(reduced_matrix_form)
    x = [1 if c  == 'real' else 0 for c in data['label'].to_list()]
    print(f1_score(predictions, x))
            
if __name__ == "__main__":
    #export()
    fit()