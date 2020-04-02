import numpy as np

class SGD(object):

    def __init__(self,lr:float=0.01):
        self.lr = lr

    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
