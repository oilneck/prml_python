import numpy as np
from deepL_module.nn.layers.layer import *
from deepL_module.nn.cost_functions import *
from deepL_module.nn.optimizers import *
from collections import OrderedDict

class Neural_net(object):

    def __init__(self, n_input, n_hidden, n_output, weight_std:float = 0.01, alpha:float = 0.):
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_size_list = n_hidden
        self.hidden_layer_num = len(self.hidden_size_list)
        self.alpha = alpha # Weight decay coe.
        self.params = {}

        self.__init_weight(weight_std)

        self.layers = OrderedDict()

        self.cost_function = None


    def __init_weight(self, weight_std):
        all_size_list = [self.n_input] + self.hidden_size_list + [self.n_output]
        for idx in range(1, len(all_size_list)):
            self.params['W' + str(idx)] = weight_std * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def add(self,layer:list):
        for n,key in enumerate(layer,1):
            arg = [self.params['W' + str(n)],self.params['b' + str(n)]]
            self.layers['layer' + str(n)] = eval(key.capitalize() + '_Layer' + '(*arg)')

    def set_loss(self,name:str='sum_squared_error'):
        if name == 'sum_squared_error':
            self.cost_function = Sum_squared_error()

        elif name == 'categorical_crossentropy':
            self.cost_function = Softmax_cross_entropy()

        else:
            raise KeyError("Not exist cost function name : {}".format(name))

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):

        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.alpha * np.sum(np.square(W))

        return self.cost_function(y, t) + weight_decay


    def gradient(self, x, t):

        # forward
        self.loss(x, t)

        # backward
        dout = self.cost_function.delta()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['layer' + str(idx)].dW + self.alpha * self.layers['layer' + str(idx)].W
            grads['b' + str(idx)] = self.layers['layer' + str(idx)].db

        return grads
