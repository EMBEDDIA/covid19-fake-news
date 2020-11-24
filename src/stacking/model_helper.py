# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 14:34:06 2020

@author: Bosec
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class ColumnarDataset(Dataset):
   def __init__(self, df, y):
       self.x = df
       self.y = y
        
   def __len__(self): 
       return len(self.y)
    
   def __getitem__(self, idx):
       row = self.x.iloc[idx, :].to_list()      
       return np.array(row), self.y[idx] 
   
class ShallowNet(nn.Module):    
  def __init__(self, n_features, p = 1.00):
    super(ShallowNet, self).__init__()
    self.a1 = nn.Linear(n_features, 2)
   
  def forward(self, x):
    return torch.sigmoid(self.a1(x))



class TwoNet(nn.Module):    
  def __init__(self, n_features, embedding_dim = 256):
    super(TwoNet, self).__init__()
    self.a1 = nn.Linear(n_features, embedding_dim)
    self.a2 = nn.Linear(embedding_dim, 2)
   
  def forward(self, x):
    x = torch.relu(self.a1(x))
    return torch.sigmoid(self.a2(x))



class ThreeNet(nn.Module):    
  def __init__(self, n_features, e1 = 1024, e2 = 896, e3 = 640, e4 = 512, e5=216, p = 0.4):
    super(ThreeNet, self).__init__()
    self.a1 = nn.Linear(n_features, e2)
    self.a2 = nn.Linear(e2, e3)
    self.a3 = nn.Linear(e3, e4)
    self.a4 = nn.Linear(e4,2)
    self.dropout = nn.Dropout(p) 
    
  def forward(self, x):
    x = F.selu(self.dropout(self.a1(x)))
    x = F.selu(self.dropout(self.a2(x)))
    x = F.selu(self.dropout(self.a3(x)))
    x = torch.sigmoid(self.a4(x))
    return x




class FiveNet(nn.Module):    
  def __init__(self, n_features, e1 = 1024, e2 = 896, e3 = 640, e4 = 512, e5=216, p = 0.4):
    super(FiveNet, self).__init__()
    self.a1 = nn.Linear(n_features, e2)
    self.a2 = nn.Linear(e2, e3)
    self.a3 = nn.Linear(e3, e4)
    self.a4 = nn.Linear(e4, e5)
    self.a5 = nn.Linear(e5,2)
    self.dropout = nn.Dropout(p) 
    
  def forward(self, x):
    x = F.selu(self.dropout(self.a1(x)))
    x = F.selu(self.dropout(self.a2(x)))
    x = F.selu(self.dropout(self.a3(x)))
    x = F.selu(self.dropout(self.a4(x)))
    x = torch.sigmoid(self.a5(x))
    return x

