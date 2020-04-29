import numpy as np

class Batch_norm_Layer():

    def __init__(self, gamma, beta, momentum:float = 0.9):
        self.gamma = gamma
        self.beta = beta
        self.moment = momentum
        self.n_param = 0
        self.dgamma = None
        self.dbeta = None
        self.out = None
        self.moving_mean = None
        self.moving_var = None


    def forward(self, x, is_training:bool):

        if self.moving_mean is None:
            N, D = x.shape
            self.moving_mean = np.zeros(D)
            self.moving_var = np.zeros(D)
            self.n_param = 4 * D

        if is_training:
            n_sample = x.shape[0]
            mu = np.mean(x, axis=0)
            x_mu = x - mu
            var = np.var(x, axis=0)

            X_norm = x_mu / np.sqrt(var + 1e-8)

            self.cache = (n_sample, x_mu, var, X_norm)

            self.moving_mean = self.moment * self.moving_mean + (1-self.moment) * mu
            self.moving_var = self.moment * self.moving_var + (1-self.moment) * var
        else:
            x_mu = x - self.moving_mean
            X_norm = x_mu / np.sqrt(self.moving_var + 1e-8)

        out = self.gamma * X_norm + self.beta
        self.out = out
        return out

    def backward(self, delta):
        N, x_mu, var, X_norm = self.cache

        std_inv = 1. / np.sqrt(var + 1e-8)

        dX_norm = delta * self.gamma
        dvar = -0.5 * np.sum(dX_norm * x_mu, axis=0) * np.power(std_inv, 3)
        dmu = np.sum(dX_norm * -std_inv, axis=0) \
            + dvar * np.mean(-2. * x_mu, axis=0)

        dx = (dX_norm * std_inv) + (2. * dvar * x_mu / N) + (dmu / N)

        # update param beta & gamma
        self.dgamma = np.sum(delta * X_norm, axis=0)
        self.dbeta = np.sum(delta, axis=0)

        return dx
