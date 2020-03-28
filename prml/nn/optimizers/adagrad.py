import numpy as np
from nn.feedforward_nn import Feed_Forward
class Adagrad(object):

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
        eps = self.epsilon
        lr = self.learning_rate
        h = 0
        for t in range(1,self.n_iter):
            grad = self.nn.gradE(w)[0]
            h += grad * grad
            eta = lr / (eps + np.sqrt(h))
            w -= eta * grad
        return(w)
