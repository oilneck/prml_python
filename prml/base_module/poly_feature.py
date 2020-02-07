import numpy as np
class Poly_Feature(object):

    def __init__(self,degree=9):
        self.M = degree


    def polynomial_feature(self,train_x:np.ndarray):
        return np.array(([train_x]*(self.M + 1))).T**(np.array([np.arange(0,self.M + 1)]*(train_x.shape[0])))


    def transform(self,train_x:np.ndarray):
        if train_x.ndim == 1:
            self.PHI = self.polynomial_feature(train_x)
        else:
            X = np.zeros((train_x.shape[0],1))
            for m in range(self.M + 1):
                X = np.append(X,train_x**m,axis=1)
            self.PHI = X[:,train_x.shape[1]:]
        return self.PHI