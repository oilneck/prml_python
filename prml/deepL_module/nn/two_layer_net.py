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

        self.cost_function = Sum_squared_error()

    def add(self,layer:list=['tanh','linear']):
        for n,key in enumerate(layer,1):
            arg = [self.params['W'+str(n)],self.params['b'+str(n)]]
            self.layers['layer'+str(n)] = eval(key.capitalize() + '_Layer' + '(*arg)')

    def set_loss(self,name:str='sum_squared_error'):
        if name == 'sum_squared_error':
            self.cost_function = Sum_squared_error()
        elif name == 'categorical_crossentropy':
            self.cost_function = Softmax_cross_entropy()
        else:
            raise KeyError("not exist cost function")

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
