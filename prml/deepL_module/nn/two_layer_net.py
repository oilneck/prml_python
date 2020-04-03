import numpy as np
from deepL_module.nn.layers.layer import *
from deepL_module.nn.cost_functions import *
from deepL_module.nn.optimizers import *
from collections import OrderedDict

class Two_layer_net(object):

    def __init__(self,n_input:int=1,n_hidden:int=3,n_output:int=1):
        self.params = {}
        self.params['W1'] = np.random.random((n_input,n_hidden))
        self.params['b1'] = np.zeros(n_hidden)
        self.params['W2'] = np.random.random((n_hidden,n_output))
        self.params['b2'] = np.zeros(n_output)

        self.layers = OrderedDict()
        self.layers['layer1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['tanh'] = Tanh_Layer()
        self.layers['layer2'] = Affine(self.params['W2'],self.params['b2'])
        self.layers['identity'] = Linear_Layer()

        self.cost_function = Sum_squared_error()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self,x,t):
        y = self.predict(x)
        return self.cost_function(y,t)

    def gradient(self,x,t):
        # forward
        self.loss(x,t)

        # backward
        dout = self.cost_function.delta()


        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['layer1'].dW
        grads['b1'] = self.layers['layer1'].db
        grads['W2'] = self.layers['layer2'].dW
        grads['b2'] = self.layers['layer2'].db

        return grads
