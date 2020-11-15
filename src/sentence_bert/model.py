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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch.optim as optim
from tqdm import tqdm

import model_helper 

def prepare_loaders(train_dataset, val_dataset, batch_size=20):
    torch.manual_seed(1903)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader 


def evaluate_env(net, test_loader, device=torch.device("cuda"), mode = "VALID"):
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
    print()
    print(f'{mode} accuracy: {accuracy_score(orgs, preds)}')
    print(f'{mode} F1-score: {f1_score(orgs, preds)}')
    print(f'{mode} precision: {precision_score(orgs, preds)}')
    print(f'{mode} recall: {recall_score(orgs, preds)}')
    return orgs, preds
    
def predict(test_dataset, net, batch_size=32):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    evaluate_env(net, test_loader, mode="TEST")
    
    
def train_NN(train_dataset, val_dataset , max_epochs = 1000, batch_size = 300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader, val_loader = prepare_loaders(train_dataset, val_dataset, batch_size)

    net = model_helper.ShallowNet(768)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)                                          
         
    scores = {}
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
                          
        o, p = evaluate_env(net, val_loader, device) 
        score = f1_score(o, p)
        scores[score] = net
    
    return scores[max(list(scores.keys()))]
