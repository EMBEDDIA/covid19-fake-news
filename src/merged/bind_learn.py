# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:11:14 2020

@author: Bosec
"""
import os
import numpy as np
import pandas as pd
import parse_data
import config
import sentence_transfomers_b
import lsa_model
import statistical_features
import tax2vec_model
from model import train_NN, predict
from model_helper import ColumnarDataset
from sklearn import preprocessing


def prepare_texts_5net(texts):
    bert_features = sentence_transfomers_b.fit_space(texts)
    lsa_features = lsa_model.fit_space(texts)
    stat_features = statistical_features.fit_space(texts)    
    bert_lsa = np.hstack((bert_features, lsa_features))
    bl_stats = np.hstack((bert_lsa, stat_features))
    centered =  preprocessing.scale(bl_stats)
    print(np.std(centered),np.mean(centered))
    df = pd.DataFrame(centered)    
    return df

def prepare_texts_berts(texts):
    bert_features = sentence_transfomers_b.fit_space(texts, model="distilbert-base-nli-mean-tokens")
    roberta_features = sentence_transfomers_b.fit_space(texts, model="roberta-large-nli-stsb-mean-tokens")
    xml_features =  sentence_transfomers_b.fit_space(texts, model="xlm-roberta-base")
    stat_features = statistical_features.fit_space(texts)    
    bert_lsa = np.hstack((bert_features, roberta_features))
    bl_stats = np.hstack((bert_lsa, xml_features))
    final = np.hstack((bl_stats, stat_features))
    centered =  preprocessing.scale(final)
    print(np.std(centered),np.mean(centered))
    df = pd.DataFrame(centered)    
    return df

def prepare_main(texts, combo = ["dbert", "tax2vec", "lsa", "statistical"]):
    spaces = {
              "dbert" : sentence_transfomers_b.fit_space(texts, model="distilbert-base-nli-mean-tokens"),
              "roberta" : sentence_transfomers_b.fit_space(texts, model="roberta-large-nli-stsb-mean-tokens"),
              "xml" : sentence_transfomers_b.fit_space(texts, model="xlm-roberta-base"),
              "statistical" : statistical_features.fit_space(texts),
              "tax2vec" : tax2vec_model.fit_space(texts),
              "lsa" : lsa_model.fit_space(texts)
              }
    final_features = None
    for comb in combo:
        print(f"Prepearing space {comb}")
        new_features = spaces[comb]        
        if not final_features:
            final_features = new_features
        else:
            final_features = np.hstack((final_features, new_features))            
    return final_features


def prepare_dataset(data, prep = prepare_texts_5net):
    train_texts = data["text_a"].to_list()
    train_y = data['label'].to_list()
    train_matrix = prep(train_texts) 
    
    #del train_texts
    #train_matrix = pd.DataFrame(train_matrix)
    print(train_matrix)
    train_dataset = ColumnarDataset(train_matrix, train_y)
    return train_dataset, train_matrix.shape[1]      

import torch 
def train_nets(train_data=parse_data.get_train(), dev_data = parse_data.get_dev(), test_data = parse_data.get_test()):
    train_dataset, dims = prepare_dataset(train_data, prep = prepare_main)
    valid_dataset, dims = prepare_dataset(dev_data, prep = prepare_main)   
    test_dataset, dims = prepare_dataset(test_data, prep = prepare_main)
    for lr in [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        for p in [0.1, 0.3, 0.5, 0.7]:
            print(lr, p)
            net = train_NN(train_dataset, valid_dataset, dims ,max_epochs=100, batch_size = 32, lr = lr, dropout = p)
            predict(test_dataset, net)
            torch.save(net.state_dict(), os.path.join("pickles","t2v_bert_lsa_"+str(lr)+"_"+str(p)+".pymodel"))

            
train_nets()