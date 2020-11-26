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
import seaborn as sns

import matplotlib.pyplot as plt

import model_helper 

def prepare_loaders(train_dataset, val_dataset, batch_size=20):
    #torch.manual_seed(1903)
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
    outs = { "mode" : mode,
            "accuracy" : accuracy_score(orgs, preds),
            "F1-score" : f1_score(orgs, preds),
            "precision" : precision_score(orgs, preds),
            "recall": recall_score(orgs, preds)}
    
    if not mode == "SILENT":
        print(f'{mode} accuracy: {accuracy_score(orgs, preds)}')
        print(f'{mode} F1-score: {f1_score(orgs, preds)}')
        print(f'{mode} precision: {precision_score(orgs, preds)}')
        print(f'{mode} recall: {recall_score(orgs, preds)}')
        return orgs, preds, outs
    return f1_score(orgs, preds)    
def predict(test_dataset, net, batch_size=32):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    o, p, _ = evaluate_env(net, test_loader, mode="TEST")
    score = f1_score(o, p)
    return score
def train_NN(train_dataset, val_dataset, test_dataset="f", dims = 1552, max_epochs = 100, batch_size = 300, lr = 0.005, dropout = 0.5):
    plt.clf()
    plt.figure()
 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader, val_loader = prepare_loaders(train_dataset, val_dataset, batch_size)
    _, test_loader = prepare_loaders(train_dataset, test_dataset, batch_size)
    net = model_helper.ThreeNet(dims , p = dropout)
    print(net)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)                                          
    
    results = []
    dics = []
    scores = {}
    scores_plt = { "epochs" : list(range(max_epochs)), "validation" : [], "train" : [] , "test":  []}
    for epoch in tqdm(range(max_epochs), total = max_epochs):  
        #running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs, labels =  inputs.to(device), labels.to(device)
            optimizer.zero_grad()
    
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        o, p, dic = evaluate_env(net, train_loader, device, mode="TRAIN")
        score = f1_score(o, p)
        scores_plt["train"].append(score)        
        
        o, p, dic = evaluate_env(net, val_loader, device, mode="VALID")
        score = f1_score(o, p)
        scores_plt["validation"].append(score)
        
        scores[score] = net
        results.append(score)
    
    best = scores[max(list(scores.keys()))]
    
    o, p, dic = evaluate_env(best, train_loader, device, mode="TRAIN")
    dics.append(dic)
    
    o, p, dic = evaluate_env(best, val_loader, device, mode="VALID")
    dics.append(dic)
    
    o, p, dic = evaluate_env(best, test_loader, device, mode="TEST")
    dics.append(dic)
    
    df = pd.DataFrame(dics)
    name = "3net"+str(lr)+"_prop_"+str(dropout)
    df.to_csv("log_data/3net/"+name+".csv",index=False)
    import pickle
    with open("log_data/3net/"+name+".pkl", 'wb') as f:
        pickle.dump(df, f)
        
    return best
