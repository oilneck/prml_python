import numpy as np

class Multiple_Regression(object):

    def __init__(self,alpha=0):
        self.alpha = alpha
        self.weight_vector = None

    def fit(self,PHI:np.ndarray,train_y:np.ndarray):
        tilde_A = PHI.T @ PHI + self.alpha * np.eye(PHI.shape[1],PHI.shape[1])
        self.weight_vector = np.linalg.solve(tilde_A,PHI.T @ train_y)

    def predict(self,test_PHI:np.ndarray):
        return np.dot(test_PHI,self.weight_vector)
