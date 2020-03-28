import numpy as np
from nn.feedforward_nn import Feed_Forward
class Momentum(object):

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

        if 'momentum' in param.keys():
            self.gamma = param['momentum']
        else:
            self.gamma = 0.9

        if 'clipnorm' in param.keys():
            self.clipnorm = param['clipnorm']
        else:
            self.clipnorm = 0.5

    def set_train_data(self,x:np.array,t:np.array):
        self.nn.xlist = x
        self.nn.tlist = t

    def normalize(self,grad):
        return grad * self.clipnorm / np.linalg.norm(grad)

    def update(self,x,t,w,**kwargs):
        self.set_train_data(x,t)
        self.set_param(kwargs)
        gamma = self.gamma
        lr = self.learning_rate
        v = np.zeros_like(w)
        for t in range(1,self.n_iter):
            grad = self.nn.gradE(w)[0]
            v = gamma * v - lr * self.normalize(grad)
            w += v
        return(w)
