import numpy as np
from deepL_module.nn.layers import *
from deepL_module.nn.cost_functions import *
from deepL_module.nn.optimizers import *
from deepL_module.nn.metrics import *
from deepL_module.base import *
from collections import OrderedDict

class Neural_net(object):

    def __init__(self, n_input, n_hidden, n_output, weight_std:float = None, alpha:float = 0.):
        if isinstance(n_hidden,int): n_hidden = [n_hidden]
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden_list = n_hidden
        self.total_hidden_num = len(self.n_hidden_list)
        self.alpha = alpha # Weight decay coefficient.
        self.params = {}

        self.__init_weight(weight_std)

        self.layers = OrderedDict()

        self.cost_function = None
        self.metric = None
        self.optim = None


    def __init_weight(self, weight_std):
        node_num_list = [self.n_input] + self.n_hidden_list + [self.n_output]

        for idx in range(1, len(node_num_list)):

            if weight_std is None:
                scale = np.sqrt(1. / node_num_list[idx - 1])

            elif isinstance(weight_std,(int,float,np.number)):
                scale = weight_std

            else:
                raise TypeError("weight_std must be float type")

            self.params['W' + str(idx)] = scale * np.random.randn(node_num_list[idx-1], node_num_list[idx])
            self.params['b' + str(idx)] = np.zeros(node_num_list[idx])

    def __call__(self,X):
        return self.predict(X)

    def add(self,layer:list):
        n_layer = self.total_hidden_num + 1
        assert len(layer) == n_layer, \
        'The number of layers must be {} layers'.format(str(n_layer))

        for n,key in enumerate(layer,1):
            arg = [self.params['W' + str(n)],self.params['b' + str(n)]]
            self.layers['layer' + str(n)] = eval(key.capitalize() + '_Layer' + '(*arg)')


    def set_loss(self,name:str='sum_squared_error'):
        loss_comp = False

        if name == 'sum_squared_error':
            self.cost_function = Sum_squared_error()
            self.metric = Regression_accuracy()
            loss_comp = True

        elif name == 'categorical_crossentropy':
            self.cost_function = Softmax_cross_entropy()
            self.metric = Categorical_accuracy()
            loss_comp = True

        elif name == 'binary_crossentropy':
            self.cost_function = Sigmoid_cross_entropy()
            self.metric = Binary_accuracy()
            loss_comp = True

        else:
            raise KeyError("Not exist cost function name : {}".format(name))

        return loss_comp

    def optimizer(self,method):
        opt_dict = {'sgd':SGD(),'rmsprop':RMSprop(),'momentum':Momentum(),
                    'adam':Adam(),'adagrad':Adagrad(),'adadelta':Adadelta()}
        optim_comp = False

        if isinstance(method,str):
            for key in opt_dict.keys():
                if key == method:
                    self.optim = opt_dict[method]
                    optim_comp = True
            if optim_comp is not True:
                raise KeyError("Not exist optimizer name : {}".format(method))

        for val in opt_dict.values():
            if isinstance(method,type(val)):
                self.optim = method
                optim_comp = True

        if optim_comp is not True:
            raise ValueError('Could not interpret optimizer: ' + str(method))

        return optim_comp

    def compile(self,loss:str, optimizer):

        opt_compiled = self.optimizer(optimizer)
        loss_compiled = self.set_loss(loss)
        _is_compiled = opt_compiled and loss_compiled

        return _is_compiled

    def predict(self,X):
        y = self.feed_forward(X)
        return self.cost_function.activate(y)

    def feed_forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def accuracy(self, x, t):
        y = self.predict(x)
        return self.metric.accuracy(y,t)

    def loss(self, x, t):

        y = self.feed_forward(x)

        weight_decay = 0
        for idx in range(1, self.total_hidden_num + 2):
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
        for idx in range(1, self.total_hidden_num+2):
            grads['W' + str(idx)] = self.layers['layer' + str(idx)].dW + self.alpha * self.layers['layer' + str(idx)].W
            grads['b' + str(idx)] = self.layers['layer' + str(idx)].db

        return grads

    def fit(self, X_train:np.ndarray, t_train:np.ndarray, n_iter=1000, batch_size=None, history:bool=False):

        hist = {}
        hist['loss'] = []
        hist['acc'] = []

        for _ in range(int(n_iter)):
            x_batch, t_batch = get_mini_batch(X_train, t_train, batch_size)
            grads = self.gradient(x_batch, t_batch)
            self.optim.update(self.params, grads)

            if history:
                loss = self.loss(x_batch, t_batch)
                score = self.accuracy(x_batch, t_batch)
                hist['loss'].append(loss)
                hist['acc'].append(score)

            else:
                hist = None

        return hist
