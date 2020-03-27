import numpy as np
from nn.feedforward_nn import Feed_Forward
class Adam(object):

    def __init__(self,n_in,n_hid,n_out):
        self.nn = Feed_Forward(n_in,n_hid,n_out)


    def set_param(self,param):
        if 'learning_rate' in param.keys():
            self.learning_rate = param['learning_rate']
        else:
            self.learning_rate = 0.3

        if 'n_iter' in param.keys():
            self.n_iter = param['n_iter']
        else:
            self.n_iter = int(1500)

        if 'beta_1' in param.keys():
            self.beta_1 = param['beta_1']
        else:
            self.beta_1 = 0.95

        if 'beta_2' in param.keys():
            self.beta_2 = param['beta_2']
        else:
            self.beta_2 = 0.95

        if 'epsilon' in param.keys():
            self.epsilon = param['epsilon']
        else:
            self.epsilon = 1e-8

    def set_train_data(self,x:np.array,t:np.array):
        self.nn.xlist = x
        self.nn.tlist = t

    def update(self,x,t,w_vec,**kwargs):
        self.set_train_data(x,t)
        self.set_param(kwargs)
        beta_1,beta_2 = self.beta_1,self.beta_2
        epsilon = self.epsilon
        m = 0
        v = 0
        for t in range(1,self.n_iter):
            [gradE,E] = self.nn.gradE(w_vec)
            g = gradE
            m = beta_1 * m + (1 - beta_1) * g
            v = beta_2 * v + (1 - beta_2) * g * g
            hat_m = m / (1 - beta_1 ** t)
            hat_v = v / (1 - beta_2 ** t)
            w_vec -= self.learning_rate * hat_m / (epsilon + np.sqrt(hat_v))
        return(w_vec)
