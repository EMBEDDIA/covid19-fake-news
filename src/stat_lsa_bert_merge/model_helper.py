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
  def __init__(self, n_features):
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