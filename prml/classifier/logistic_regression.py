import numpy as np
class Logistic_Regression(object):

    @staticmethod
    def _sigmoid(a):
        return 1 / (1+np.exp(-a))

    #Calculate new weight -> INPUT:feature-vector, target-vector, w_old  OUTPUT:w_new
    def make_NEW_weight(self,PHI,t,w_old):
        #PHI,R,H -> design matrix, weighted matrix, Hessian matrix, respectively.
        y = self._sigmoid(w_old @ PHI.T)
        R = np.diag(y * (1-y)) + 1e-4 * np.eye(len(y), len(y))
        H = PHI.T @ (R @ PHI)
        invR = np.linalg.inv(R)
        z = PHI @ w_old - invR @ (y-t)
        return np.linalg.inv(H) @ PHI.T @ R @ z

    def fit(self, X:np.ndarray, train_t:np.ndarray):
        w = np.ones(np.size(X, 1)) * 1e-4
        while True: # Updating weight vector
            w_new = self.make_NEW_weight(X, train_t, w)
            if np.allclose(w, w_new, rtol=0.01): break
            w = w_new
        self.w = w


    def predict(self, PHI_test:np.ndarray):
        return self.w @ PHI_test.T
