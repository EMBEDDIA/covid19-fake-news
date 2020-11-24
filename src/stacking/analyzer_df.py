# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:22:39 2020

@author: Bosec
"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

idx = []
for i in range(100):
    idx.append(i)
    idx.append(i)
    idx.append(i)
    
for file in os.listdir("log_data/dfs"):
    df = pd.read_csv(os.path.join("log_data/dfs",file))    
    df["epochs"] = idx
    for t in ["TRAIN", "VALID", "TEST"]:
        train = df.loc[df["mode"] == t] 
        del train["mode"]
        print(train)
        plt.title(t)
        plt.ylabel('f1_score')
        plot = sns.lineplot(x='epochs', y='value', hue='variable', data=pd.melt(train, ['epochs']))
        plot.figure.savefig("_".join(["imgs/", t, file[1:-4], ".png"]), dpi=300)
        plt.clf()
        