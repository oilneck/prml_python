import numpy as np

class BoltzmannMachine(object):


    def __init__(self, n_unit:int, alpha:float=0.1):
        self.n_unit = n_unit
        self.alpha = alpha
        self.params = {}
        self.params['W'] = np.random.random((n_unit, n_unit))
        self.params['b'] = np.random.random(n_unit)
        self.errors = []


    def calc_grad(self, X:np.ndarray):
        data_ave = X.T @ X
        tmp = np.tanh(X.dot(self.params['W'].T) + self.params['b'])
        model_ave = np.dot(tmp.T, X)
        dW = (data_ave - model_ave) / np.size(X, 0)
        db = np.mean(X - tmp, axis=0)
        return {'W': dW + self.alpha * self.params['W'], 'b':db}


    def update(self, params:dict, grads:dict, lr:float=1):
        for key in params.keys():
            params[key] += lr * grads[key]


    def fit(self, X_train:np.ndarray, n_iter:int=100, lr:float=1):

        errors = []
        for _ in range(int(n_iter)):
            err = self.calc_energy(X_train)
            errors.append(err)
            self.update(self.params, self.calc_grad(X_train), lr=lr)
            if np.allclose(err, self.calc_energy(X_train)):
                break

        self.errors = errors.copy()
        return errors


    def calc_energy(self, X:np.ndarray):
        H = np.trace(X @ self.params['W'] @ X.T)
        H += (X @ self.params['b']).sum(axis=0)
        tmp = np.log(np.cosh(X @ self.params['W'].T + self.params['b'])).sum()
        err = -(H - tmp) / X.shape[0]
        return err
