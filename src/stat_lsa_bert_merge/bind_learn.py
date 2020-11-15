# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:11:14 2020

@author: Bosec
"""

import numpy as np
import pandas as pd
import parse_data
import config
import sentence_transfomers_b
import lsa_model
import statistical_features
from model import train_NN, predict

from sklearn import preprocessing
def prepare_texts(texts):
    bert_features = sentence_transfomers_b.fit_space(texts)
    lsa_features = lsa_model.fit_space(texts)
    stat_features = statistical_features.fit_space(texts)    
    bert_lsa = np.hstack((bert_features, lsa_features))
    bl_stats = np.hstack((bert_lsa, stat_features))
    centered =  preprocessing.scale(bl_stats)
    df = pd.DataFrame(centered)    
    return df



 #Train the dev data
#dev_texts = dev_data["text_a"].to_list()
#dev_y = dev_data['label'].to_list()

import pandas as pd
from model_helper import ColumnarDataset

def prepare_dataset(data):
    train_texts = data["text_a"].to_list()
    train_y = data['label'].to_list()
    train_matrix = prepare_texts(train_texts) 
    del train_texts
    train_matrix = pd.DataFrame(train_matrix)
    print(train_matrix)
    train_dataset = ColumnarDataset(train_matrix, train_y)
    return train_dataset      
    
def train_nets(train_data=parse_data.get_train(), dev_data = parse_data.get_dev(), test_data = parse_data.get_test()):
    train_dataset = prepare_dataset(train_data)
    valid_dataset = prepare_dataset(dev_data)
    test_dataset = prepare_dataset(test_data)
    net = train_NN(train_dataset, valid_dataset)
    
    predict(test_dataset, net)
    
train_nets()