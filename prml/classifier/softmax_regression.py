import numpy as np
class Softmax_Regression(object):

    @staticmethod
    def _softmax(X):
        MAX_val = np.max(X, axis=-1, keepdims=True)
        return np.exp(X-MAX_val)/np.sum(np.exp(X-MAX_val), axis = 1, keepdims = True)

    def Hessian(self,PHI,W,cls_k:int=0):
        Y = self._softmax(PHI @ W)
        R = np.diag((Y[:,cls_k] * (1-Y[:,cls_k])))
        H = PHI.T @ R @ PHI
        return H + 0.001 * np.eye(len(H),len(H))

    def fit(self,train_PHI,train_t):
        W = np.zeros((np.size(train_PHI, 1), np.size(train_t, 1)))
        while True: #Updating weight matrix
            Y = self._softmax(train_PHI @ W)
            W_NEW = np.zeros((np.size(train_PHI, 1), np.size(train_t, 1)))
            for cls_num in range(np.size(train_t, 1)):
                W_NEW[:,cls_num] = W[:,cls_num] - (np.linalg.inv(self.Hessian(train_PHI,W,cls_num)) @ train_PHI.T @ (Y-train_t))[:,cls_num]
            if np.allclose(W, W_NEW,rtol=0.01): break
            W = W_NEW
        self.W = W

    def predict(self,test_PHI:np.ndarray):
        return np.argmax(self._softmax(test_PHI @ self.W), axis=1)
