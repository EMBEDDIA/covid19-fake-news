# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:02:22 2020

@author: Bosec
"""
import os
import numpy as np
import parse_data
import pandas as pd
import lsa_model
import statistical_features
from model import train_NN, predict
from model_helper import ColumnarDataset
from sklearn import preprocessing
import sentence_transfomers
import statistical_baseline
import torch 

text = "ncludes high-quality download in MP3, FLAC and more. Paying supporters also get unlimited streaming via the free Bandcamp app."


def get_features(texts):
    lsa_features = lsa_model.fit(texts).reshape((len(texts), 1))
    bert_features = sentence_transfomers.fit(texts, "distilbert-base-nli-mean-tokens").reshape((len(texts), 1))
    stat_features = statistical_baseline.fit(texts).reshape((len(texts), 1)).reshape((len(texts), 1))
    roberta_features = sentence_transfomers.fit(texts, "roberta-large-nli-stsb-mean-tokens").reshape((len(texts), 1))
    xlm_features = sentence_transfomers.fit(texts, "xlm-r-large-en-ko-nli-ststb").reshape((len(texts), 1))    
    features = np.hstack(([lsa_features, bert_features, stat_features, roberta_features, xlm_features, roberta_features]))
    print(features.shape)
    return preprocessing.scale(features)
    
def get_features_probs(texts):
    lsa_features = lsa_model.fit_probs(texts).reshape((len(texts), 1))
    bert_features = sentence_transfomers.fit_probs(texts, "distilbert-base-nli-mean-tokens").reshape((len(texts), 1))
    stat_features = statistical_baseline.fit_probs(texts).reshape((len(texts), 1))
    roberta_features = sentence_transfomers.fit_probs(texts, "roberta-large-nli-stsb-mean-tokens").reshape((len(texts), 1))
    xlm_features = sentence_transfomers.fit_probs(texts, "xlm-r-large-en-ko-nli-ststb").reshape((len(texts), 1))    
    features = np.hstack(([lsa_features, bert_features, stat_features, roberta_features, xlm_features, roberta_features]))
    print(features.shape)
    return preprocessing.scale(features)

def prepare_dataset(data, prep = get_features_probs):
    train_texts = data["text_a"].to_list()
    train_y = data['label'].to_list()
    train_matrix = prep(train_texts)     
    train_matrix = pd.DataFrame(train_matrix)
    print(train_matrix)
    train_dataset = ColumnarDataset(train_matrix, train_y)
    return train_dataset, train_matrix.shape[1]      

def train_nets(train_data=parse_data.get_train(), dev_data = parse_data.get_dev(), test_data = parse_data.get_test()):
    train_dataset, dims = prepare_dataset(train_data, prep = get_features_probs)
    valid_dataset, dims = prepare_dataset(dev_data, prep = get_features_probs)   
    test_dataset, dims = prepare_dataset(test_data, prep = get_features_probs)
    for lr in [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        for p in [0.1, 0.3, 0.5, 0.7]:
            print(lr, p)
            net = train_NN(train_dataset, valid_dataset, dims ,max_epochs=100, batch_size = 32, lr = lr, dropout = p)
            predict(test_dataset, net)
            torch.save(net.state_dict(), os.path.join("pickles","t2v_bert_lsa_"+str(lr)+"_"+str(p)+".pymodel"))

            
#train_nets()
data = parse_data.get_dev()
train_texts = data["text_a"].to_list()
train_y = data['label'].to_list()
print(get_features(train_texts))
#print(get_features_probs([text]))

