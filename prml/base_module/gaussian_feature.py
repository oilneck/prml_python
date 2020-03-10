import numpy as np
class Gaussian_Feature(object):

    def __init__(self,mean:np.array=np.linspace(-1, 1, 11),variance:float=0.1):
        self.mean = mean
        self.var = variance

    def gauss_function(self,x,mean):
        return np.exp(-0.5 * np.square(x - mean) / self.var)

    def transform(self,train_x:np.ndarray):
        X = np.ones((train_x.shape[0],1))
        for m in self.mean:
            X = np.append(X,self.gauss_function(train_x,m),axis=1)
        self.PHI = np.array(X)
        return self.PHI
