import config 
import numpy
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

def get_train(path = config.TRAIN):
    df = pd.read_csv(path, sep="\t")
    return df
    
def get_dev(path = config.DEV):
    df = pd.read_csv(path, sep="\t")
    return df
    
def get_test(path = config.TEST):
    df = pd.read_csv(path, sep="\t")
    return df
    
def readTrain(path = config.PATH_TRAIN):
    df = pd.read_csv(path)
    del df["id"]
    return df
    

def readValidation(path = config.PATH_VALID):
    df = pd.read_csv(path)
    del df["id"]
    print(df)
    return df

    

if __name__ == "__main__":
    #readTrain()
    #readValidation()
    print(get_train())