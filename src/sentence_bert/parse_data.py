import config 
import numpy
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def readTrain(path = config.PATH_TRAIN, skiprows = 0):
    df = pd.read_csv(path, skiprows=skiprows)
    del df["id"]
    return df
    

def readValidation(path = config.PATH_VALID):
    df = pd.read_csv(path)
    del df["id"]
    print(df)
    return df

    

if __name__ == "__main__":
    readTrain()
    readValidation()