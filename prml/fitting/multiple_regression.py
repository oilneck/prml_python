import numpy as np
from base_module.poly_feature import Poly_Feature

class Multiple_Regression(object):
    
    def __init__(self,degree=9,lamda=0):
        self.M = degree
        self.lamda = lamda

    def fit(self,train_x:np.ndarray,train_y:np.ndarray):
        feature = Poly_Feature(self.M)
        PHI = feature.transform(train_x)
        tilde_A = PHI.T @ PHI + self.lamda * np.eye(PHI.shape[1],PHI.shape[1])
        self.weight_vector = np.linalg.solve(tilde_A,PHI.T @ train_y)[::-1]

    def predict(self,test_x):
        return np.poly1d(self.weight_vector)(test_x)
