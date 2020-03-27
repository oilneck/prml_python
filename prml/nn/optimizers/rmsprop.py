import numpy as np
from nn.feedforward_nn import Feed_Forward
class RMSprop(object):

    def __init__(self,n_in,n_hid,n_out,regularization_coe):
        self.nn = Feed_Forward(n_in,n_hid,n_out,regularization_coe)


    def set_param(self,param):
        if 'learning_rate' in param.keys():
            self.learning_rate = param['learning_rate']
        else:
            self.learning_rate = 0.01

        if 'n_iter' in param.keys():
            self.n_iter = param['n_iter']
        else:
            self.n_iter = int(1000)

        if 'rho' in param.keys():
            self.rho = param['rho']
        else:
            self.rho = 0.9

        if 'epsilon' in param.keys():
            self.epsilon = param['epsilon']
        else:
            self.epsilon = 1e-8

    def set_train_data(self,x:np.array,t:np.array):
        self.nn.xlist = x
        self.nn.tlist = t

    def update(self,x,t,w,**kwargs):
        self.set_train_data(x,t)
        self.set_param(kwargs)
        rho = self.rho
        epsilon = self.epsilon
        lr = self.learning_rate
        v = 0
        for t in range(1,self.n_iter):
            [gradE,E] = self.nn.gradE(w)
            g = gradE
            v = rho * v + (1 - rho) * g * g
            eta = lr / (epsilon + np.sqrt(v))
            w -= eta * g
        return(w)
