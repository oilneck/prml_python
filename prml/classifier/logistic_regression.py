import numpy as np
class Logistic_Regression(object):

    @staticmethod
    def _sigmoid(a):
        return 1/(1+np.exp(-a))

    #Calculate new weight -> INPUT:feature-vector, target-vector, w_old  OUTPUT:w_new
    def make_NEW_weight(self,PHI,t,w_old):
        #PHI,R,H -> design matrix, weighted matrix, Hessian matrix, respectively.
        PHI_T = PHI.T
        y = self._sigmoid(w_old @ PHI_T)
        R = np.diag(y * (1-y)) + 0.001 * np.eye(len(y),len(y))
        H = PHI_T @ (R @ PHI_T.T)
        R_INV = np.linalg.inv(R)
        z = PHI_T.T @ w_old - R_INV @ (y-t)
        return np.linalg.inv(H) @ PHI_T @ R @ z

    def fit(self,train_PHI,train_t):
        w = np.ones(np.size(train_PHI,1)) * 0.0001
        while True:
            w_new = self.make_NEW_weight(train_PHI,train_t,w)   # Updating weight vector
            if np.allclose(w, w_new,rtol=0.01): break
            w = w_new
        self.w = w


    def predict(self,test_PHI:np.ndarray):
        return self.w @ test_PHI.T
