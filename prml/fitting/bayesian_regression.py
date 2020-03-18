import numpy as np
from base_module import Poly_Feature,Gaussian_Feature
class Bayesian_Regression(object):

    def __init__(self,degree:int=9,alpha:float=0.005,beta:float=11):
        self.M = degree
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_cov = None
        self.feature = Poly_Feature(self.M)

    def fit(self,train_x:np.ndarray,t:np.ndarray):
        PHI = self.feature.transform(train_x)
        S_inv = self.beta * (PHI.T @ PHI) + self.alpha * np.eye(np.size(PHI,1),np.size(PHI,1))
        S = np.linalg.inv(S_inv)
        self.w_mean = self.beta * S @ PHI.T @ t
        self.w_cov = S

    def predict(self,test_x:np.ndarray,get_std:bool=False):
        test_PHI = self.feature.transform(test_x)
        m_x = np.sum(test_PHI * np.array([self.w_mean] * test_x.shape[0]),axis=1)
        if get_std:
            s_x = np.sqrt(self.beta**(-1) + np.sum(test_PHI * (self.w_cov @ test_PHI.T).T,axis=1))
            return m_x,s_x
        return m_x
