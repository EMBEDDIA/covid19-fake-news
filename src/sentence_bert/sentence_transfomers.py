# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 8:27:03 2020

@author: Bosec
"""


import config
import parse_data
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import feature_construction
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
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegressionCV

#BERT
class BERTTransformer():
    def __init__(self,docs, model_name = 'distilbert-base-nli-mean-tokens'):
        self.docs = list(docs.values())
        self.model_name = model_name
        self.docsNames = list(docs.keys())
        self._model = None
        self.X = None
        
    def fit(self):
        self._model = SentenceTransformer(self.model_name)
        self.X = self._model.encode(self.docs)  
   
    def to_matrix(self):
        return np.array(self.X)
    
    def _exportModel(self,path="bert_sentences.pkl"):
        pickle.dump(self.X, open(path, "wb"))
        
    def _importModel(self,path="bert_sentences.pkl"):
        self.X = pickle.load(open(path,'rb'))
        return self.X
    

    def _export(self,path="bert_transfomer.pkl"):
        pickle.dump(self._model, open(path, "wb"))

    def _import(self,path="bert_.pkl"):
        self._model = pickle.load(open(path,'rb'))
    

def preprocess_texts(docs):
    dataframe = feature_construction.build_dataframe(docs)
    return dataframe['no_stopwords']
    
def prepare_text(texts):
    preprocessed = preprocess_texts(texts)    
    x = preprocessed.values
    docs = dict(zip(list(range(len(x))), x))
    bert = BERTTransformer(docs, "xlm-roberta-base")
    bert.fit()
    x = bert.to_matrix()
    return x

def train():
    train = parse_data.readTrain()
    x = prepare_text(train['tweet'].tolist())
    y = train['label'].tolist()       
    return x, y    


data_matrix, y = train()
print(data_matrix.shape)

"""
clf1 = GridSearchCV(x, parameters, n_jobs = -1,cv = 10, verbose = True, refit = True)
scores = cross_val_score(clf1, data_matrix, y, cv=10, scoring='f1_macro', verbose = True, n_jobs = -1)

clf1 = clf1.fit(data_matrix, y)
"""

y = [1 if c  == 'real' else 0 for c in y]



clf1 = LogisticRegressionCV(cv=10, penalty='l2', fit_intercept=True, scoring='f1').fit(data_matrix, y)


print("AVG F1-score:" + str(clf1.score(data_matrix,y)))
valid = parse_data.readValidation()
y = [1 if c  == 'real' else 0 for c in valid['label'].to_list()]
data_matrix_val = prepare_text(valid['tweet'])

print("VALIDATION F1-SCORE:", f1_score(clf1.predict(data_matrix_val), y)) 
    
