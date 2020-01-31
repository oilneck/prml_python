import numpy as np
class Linear_Regression(object):

    def __init__(self,degree=9,lamda=0):
        self.M = degree
        self.lamda = lamda

    def create_feature_matrix(self,train_x,train_y):
        PHI_T = np.array([train_x]*(self.M + 1))**(np.array([np.arange(0,self.M + 1)]*(len(train_y))).T)
        return PHI_T.T

    def fit(self,train_x:np.ndarray,train_y:np.ndarray):
        PHI = self.create_feature_matrix(train_x,train_y)
        tilde_A = PHI.T @ PHI + self.lamda * np.eye(self.M + 1,self.M + 1)
        self.weight_vector = np.linalg.solve(tilde_A,PHI.T @ train_y)[::-1]