import os.path
import gzip
import pickle
import os
import numpy as np
import urllib.request
import pandas as pd

url_base = 'http://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat'
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "\\old_faithful.csv"

def download(file_name:str="old_faithful.csv"):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base, file_path)
    print("Done")

def init_faithful():
    download()
    data_row = 26
    n_row = 0
    X = []
    dataset = {}
    f = open(save_file, 'r')
    for n,line in enumerate(f):
        data = line.split()
        if n >= data_row:
            X.append([float(data[1]), float(data[2])])
    X = np.array(X)
    df = pd.DataFrame(X,columns=['x','y'])
    df.to_csv('old_faithful.csv', sep=",")





def load_faithful():
    if not os.path.exists(save_file):
        init_faithful()

    df = pd.read_csv(save_file, header=0)
    return np.array([df.x.values, df.y.values]).T
