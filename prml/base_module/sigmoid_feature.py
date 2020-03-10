import numpy as np
class Sigmoid_Feature(object):

    def __init__(self,mean:np.array=np.linspace(-1, 1, 11),std:float=1):
        self.mean = mean
        self.std = std

    def sigmoid_function(self,x,mean):
        a = (x - mean) / self.std
        return 0.5 + 0.5 * np.tanh(0.5 * a)

    def transform(self,train_x:np.ndarray):
        if train_x.ndim == 1:
            train_x = train_x.reshape(len(train_x),1)
        X = np.ones((train_x.shape[0],1))
        for m in self.mean:
            X = np.append(X,self.sigmoid_function(train_x,m),axis=1)
        self.PHI = np.array(X)
        return self.PHI