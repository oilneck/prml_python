import numpy as np
from fitting.bayesian_regression import Bayesian_Regression

class Optimal_Bayesian_Regression(Bayesian_Regression):

    def __init__(self,degree:int=9,alpha:float=1.,beta:float=1.):
        super().__init__(degree,alpha,beta)

    def fit(self,train_x:np.ndarray,t:np.ndarray,n_iter:int=1000):
        PHI = self.feature.transform(train_x)
        eigenvalues = np.linalg.eigvalsh(self.beta * PHI.T @ PHI)
        for _ in range(n_iter):
            param = [self.alpha,self.beta]
            S_INV = self.beta * (PHI.T @ PHI) + self.alpha * np.eye(self.M+1,self.M+1)
            S = np.linalg.inv(S_INV)
            m_N = self.beta * S @ PHI.T @ t
            gamma = np.sum(eigenvalues / (self.alpha + eigenvalues))
            self.alpha = float(gamma / np.sum(m_N ** 2).clip(min=1e-5))
            self.beta = float((len(t) - gamma) / np.sum(np.square(t - PHI @ m_N)))
            if np.allclose(param,[self.alpha,self.beta]):
                break
        self.w_mean = m_N
        self.w_cov = S

    def evidence_function(self,train_x:np.ndarray,t:np.ndarray):
        PHI = self.feature.transform(train_x)
        N = len(t)
        M = np.size(PHI,1)
        Error_Value = (self.beta / 2) * np.square(t - PHI @ self.w_mean).sum() + (self.alpha/2) * np.sum(self.w_mean**2)
        return 0.5 * (N * np.log(self.beta) + M * np.log(self.alpha)  +  np.linalg.slogdet(self.w_cov)[1] -N*np.log(2*np.pi)) - Error_Value
