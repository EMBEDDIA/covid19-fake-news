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
from model import train_NN, predict
from model_helper import ColumnarDataset
from sklearn import preprocessing

def prepare_texts(texts):
    bert_features = sentence_transfomers_b.fit_space(texts)
    lsa_features = lsa_model.fit_space(texts)
    stat_features = statistical_features.fit_space(texts)    
    bert_lsa = np.hstack((bert_features, lsa_features))
    bl_stats = np.hstack((bert_lsa, stat_features))
    centered =  preprocessing.scale(bl_stats)
    print(np.std(centered),np.mean(centered))
    df = pd.DataFrame(centered)    
    return df

def prepare_dataset(data):
    train_texts = data["text_a"].to_list()
    train_y = data['label'].to_list()
    train_matrix = prepare_texts(train_texts) 
    
    #del train_texts
    #train_matrix = pd.DataFrame(train_matrix)
    print(train_matrix)
    train_dataset = ColumnarDataset(train_matrix, train_y)
    return train_dataset      

import torch 
def train_nets(train_data=parse_data.get_train(), dev_data = parse_data.get_dev(), test_data = parse_data.get_test()):
    train_dataset = prepare_dataset(train_data)
    valid_dataset = prepare_dataset(dev_data)
    test_dataset = prepare_dataset(test_data)
    for lr in [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        for p in [0.1, 0.3, 0.5, 0.7]:
            print(lr, p)
            net = train_NN(train_dataset, valid_dataset, max_epochs=100, batch_size = 32, lr = lr, dropout = p)
            predict(test_dataset, net)
            torch.save(net.state_dict(), os.path.join("pickles","net_"+str(lr)+"_"+str(p)+".pymodel"))

            
train_nets()