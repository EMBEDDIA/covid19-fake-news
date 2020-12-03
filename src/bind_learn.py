# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:11:14 2020

@author: Bosec
"""

import numpy as np
import pandas as pd

from utils import parse_data, config
from sentence_bert import sentence_transfomers_b
from lsa_baseline import lsa_model
from statistical_baseline import statistical_features

def prepare_texts(texts):
    bert_features = sentence_transfomers_b.fit_space(texts)
    lsa_features = lsa_model.fit_space(texts)
    stat_features = statistical_features.fit_space(texts)
    
    bert_lsa = np.hstack((bert_features, lsa_features))
    bl_stats = np.hstack((bert_lsa, stat_features))
    
    df = pd.DataFrame(bl_stats)
    
    return df


train_data = parse_data.get_train()
dev_data = parse_data.get_dev()
train_texts = train_data["text_a"].to_list()
train_y = train_data['label'].to_list()
train_matrix = fit_space(train_texts) 
del train_texts

 #Train the dev data
dev_texts = dev_data["text_a"].to_list()
dev_y = dev_data['label'].to_list()
dev_matrix = fit_space(dev_texts) 
del dev_texts    
