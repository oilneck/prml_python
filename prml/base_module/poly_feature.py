import numpy as np
class Poly_Feature(object):

    def __init__(self,degree=9):
        self.M = degree


    def polynomial_feature(self,train_x:np.ndarray):
        return np.array(([train_x]*(self.M + 1))).T**(np.array([np.arange(0,self.M + 1)]*(train_x.shape[0])))


    def transform(self,train_x:np.ndarray):
        if train_x.ndim == 1:
            self.PHI = self.polynomial_feature(train_x)
        elif train_x.ndim >= 2:
            col = train_x.shape[1]
            X = np.zeros((train_x.shape[0],col*(self.M + 1)))
            for m,idx in enumerate(np.arange(0,col*(self.M+1),col)):
                X[:,idx:idx+col]=train_x**m
            self.PHI = X[:,col-1:]
        return self.PHI
