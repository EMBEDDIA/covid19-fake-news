# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:08:59 2020

@author: Bosec
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transfomers import fit_space
from sklearn.metrics import f1_score as scoring_function
import torch.optim as optim
from tqdm import tqdm

from model_helper import ColumnarDataset, ShallowNet

def prepare_loaders(train_dataset, val_dataset, batch_size=20):
    torch.manual_seed(1903)
    train_loader = DataLoader(train_dataset.dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset.dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader 

def prepare_dataset(data):
    train_texts = data["text_a"].to_list()
    train_y = data['label'].to_list()
    train_matrix = fit_space(train_texts) 
    del train_texts
    train_matrix = pd.DataFrame(train_matrix)
    train_dataset = ColumnarDataset(train_matrix, train_y)
    return train_dataset      


def evaluate_env(net, test_loader, device=torch.device("cuda"), scoring_function = None):
    orgs = []
    preds = []
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels =  inputs.to(device), labels.to(device)
            outputs = net(inputs.float())
            logits = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            guessed = np.argmax(logits,axis=1)
            orgs = orgs + labels.tolist()
            preds = preds + guessed.tolist()

    if scoring_function:
        return scoring_function(orgs, preds)
    else:
        return orgs, preds
    
    
def train_NN(train, valid, max_epochs = 3, batch_size = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = prepare_dataset(train)
    val_dataset = prepare_dataset(valid)

    train_loader, val_loader = prepare_loaders(train_dataset, val_dataset, batch_size)

    net = ShallowNet(512)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)                                          
         
    #scores = {}                                          
   # scores[tmp_combo_name] = []
    
    for epoch in tqdm(range(max_epochs), total = max_epochs):  
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs, labels =  inputs.to(device), labels.to(device)
            optimizer.zero_grad()
    
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if epoch % (max_epochs/2) == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0                        
                          
        orgs, preds = evaluate_env(net, val_loader, device) 
        score = scoring_function(orgs, preds)
        print(f'Validaiton score: %d' % (score))
        #scores[score].append(score)
        
