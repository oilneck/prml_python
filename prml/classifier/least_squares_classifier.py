import numpy as np
class Least_Squares_Classifier(object):

    def fit(self,train_PHI,train_t):
        self.W = np.linalg.inv(train_PHI.T @ train_PHI) @ train_PHI.T @ train_t


    def predict(self, test_PHI:np.ndarray):
        return np.argmax(test_PHI @ self.W, axis=-1)#test_PHI @ np.diff(self.W, axis=1)
