# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:25:14 2020

@author: Bosec
"""

## some more experiments

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
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
import pickle

try:
    import umap
except:
    pass

def fit_space(X, model_path="."):
    df_final = build_dataframe(X)
    tokenizer,_,reducer = _import(lang="",path_in=model_path)
    matrix_form = tokenizer.transform(df_final)
    reduced_matrix_form = reducer.transform(matrix_form)
    return reduced_matrix_form 

def fit(X, model_path="."):
    reduced_matrix_form = fit_space(X, model_path)    
    _,clf,_ = _import(lang="",path_in=model_path)
    predictions = clf.predict(reduced_matrix_form)    
    return predictions

def evaluate(test_data=parse_data.get_test()):
    X = test_data["text_a"].to_list()
    orig = test_data['label'].to_list()
    preds = fit(X)
    print(f1_score(orig,preds))
    
def prepare_all():
    train = parse_data.get_train()
    dev = parse_data.get_dev()
    test = parse_data.get_test()
    
    X = train["text_a"].to_list() + dev["text_a"].to_list() + test["text_a"].to_list() 
    y = train["label"].to_list() + dev["label"].to_list() + test["label"].to_list() 
    
    return X,y
def train(X, final_y, output=False):

    report = []

    trained_models = {}
    
    dataframe = build_dataframe(X)   
    del X
    
    for nrep in range(1):
        for nfeat in [500, 1250, 2500, 5000, 10000, 15000, 20000]:
            for dim in [64,128, 256, 512, 768]:
                
                #"Prepare train"
                
                tokenizer, feature_names, data_matrix = get_features(dataframe, max_num_feat = nfeat, labels = final_y)
                reducer = TruncatedSVD(n_components = min(dim, nfeat * len(feature_names)-1))
                data_matrix = reducer.fit_transform(data_matrix)
                
                            
                #"Train SGD"
                logging.info("Generated {} features.".format(nfeat*len(feature_names)))
                parameters = {"loss":["hinge","log"],"penalty":["elasticnet"],"alpha":[0.01,0.001,0.0001,0.0005],"l1_ratio":[0.05,0.25,0.3,0.6,0.8,0.95],"power_t":[0.5,0.1,0.9]}
                svc = SGDClassifier()
                gs1 = GridSearchCV(svc, parameters, verbose = 0, n_jobs = 8,cv = 10, refit = True)
               
                gs1.fit(data_matrix, final_y)
                clf1 = gs1.best_estimator_
                
                scores = cross_val_score(clf1, data_matrix, final_y, cv=10, scoring='f1_macro')
                acc_svm = scores.mean()
                logging.info("TRAIN SGD 10fCV F1-score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
             
                #"Train LSA"
                parameters = {"C":[0.1,1,10,25,50,100,500],"penalty":["l2"]}
                svc = LogisticRegression(max_iter = 100000,  solver="lbfgs")
                gs2 = GridSearchCV(svc, parameters, verbose = 0, n_jobs = 8,cv = 10, refit = True)
                gs2.fit(data_matrix, final_y)
                clf2 = gs2.best_estimator_
                
                scores = cross_val_score(clf2, data_matrix, final_y, cv=10, scoring='f1_macro')
                acc_lr = scores.mean()
                logging.info("TRAIN LR 10fCV F1-score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))              
                              
                trained_models[nfeat] = ((clf1, clf2), dim)
                report.append([nfeat, acc_lr, acc_svm])
                
    
    dfx = pd.DataFrame(report)
    dfx.columns = ["Number of features","LR","SVM"]
    dfx = pd.melt(dfx, id_vars=['Number of features'], value_vars=['LR','SVM'])
    sns.lineplot(dfx['Number of features'],dfx['value'], hue = dfx["variable"], markers = True, style = dfx['variable'])
    plt.legend()
    plt.tight_layout()
    plt.savefig("CV_tfidif-merged.png",dpi = 300)
    sorted_dfx = dfx.sort_values(by = ["value"])
    print(sorted_dfx.iloc[-1,:])
    max_acc = sorted_dfx.iloc[-1,:][['Number of features','variable']]

    final_feature_number = max_acc['Number of features']
    final_learner = max_acc['variable']
    logging.info("Final feature number: {}, final learner: {}".format(final_feature_number, final_learner))
    
    index = 0 if final_learner == "SVM" else 1
    
    clf_final, dim = trained_models[final_feature_number]
    clf_final = clf_final[index]

    tokenizer, feature_names, data_matrix = get_features(dataframe, max_num_feat = final_feature_number)
    reducer = TruncatedSVD(n_components = min(dim, nfeat * len(feature_names)-1)).fit(data_matrix)
    clf_final = clf_final.fit(reducer.transform(data_matrix), final_y)
    print(hasattr(clf_final, "classes_"))
    return tokenizer, clf_final, reducer

def _import(lang='en',path_in="."):
    """Imports tokenizer,clf,reducer from param(path_in, default is ../models)"""
    tokenizer = pickle.load(open(os.path.join(config.PICKLES_PATH, "tokenizer_cv.pkl"),'rb'))
    clf = pickle.load(open(os.path.join(config.PICKLES_PATH, "clf_cv.pkl"),'rb'))
    reducer = pickle.load(open(os.path.join(config.PICKLES_PATH, "reducer_cv.pkl"),'rb'))
    return tokenizer,clf,reducer

def export():
    X, y = prepare_all()
    tokenizer, clf, reducer = train(X, y)
    with open(os.path.join(config.PICKLES_PATH, "tokenizer_cv.pkl"),mode='wb') as f:
        pickle.dump(tokenizer,f)
    with open(os.path.join(config.PICKLES_PATH, "clf_cv.pkl"),mode='wb') as f:
        pickle.dump(clf,f)
    with open(os.path.join(config.PICKLES_PATH, "reducer_cv.pkl"),mode='wb') as f:
        pickle.dump(reducer,f)

def _fit(path=""):
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
            
def evaluate_TDT():
    print("TRAIN OUTS: ")
    evaluate(parse_data.get_train())
    print("DEV OUTS: ")
    evaluate(parse_data.get_dev())
    print("TEST OUTS: ")
    evaluate(parse_data.get_test())
    
if __name__ == "__main__":
    #evaluate_TDT()
    export()