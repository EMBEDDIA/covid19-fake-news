# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:21:59 2020

@author: Bosec
"""

import pickle
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


for file in os.listdir("log_data"):
    with open(os.path.join("log_data",file),"rb") as f:
        df = pickle.load(f)
    parsed = file.split('_')
    lr = parsed[2]
    p = parsed[-1][:-4]
    print(p)
    print(df)
    plt.title('5Net SGD lr:'+str(lr)+" drop:"+str(p))
    plt.ylabel('f1_score')
    plot = sns.lineplot(x='epochs', y='value', hue='variable', data=pd.melt(df, ['epochs']))
    plot.figure.savefig("_".join(["imgs/"] + parsed[1:-1] + [".png"]), dpi=300)
    #plt.show()
    plt.clf()
    