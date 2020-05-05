import numpy as np
from deepL_module.nn.layers import *
from deepL_module.nn.layers.core import *
from deepL_module.nn.cost_functions import *
from deepL_module.nn.optimizers import *
from deepL_module.nn.metrics import *
from deepL_module.base import *
from collections import OrderedDict


class Sequential(object):

    def __init__(self, w_std:float=None, alpha:float=0.):
        self.alpha = alpha # Weight decay coefficient.
        self.wscale = w_std
        self.n_hidden = 0
        self.units_list = []
        self.params = {}

        self.layers = []

        self.cost_function = None
        self.metric = None
        self.optim = None



    def __call__(self,X):
        return self.predict(X)

    def init_W(self):

        self.n_hidden += 1
        idx = np.copy(self.n_hidden)

        scale = 1.
        if self.wscale is None:
            scale = np.sqrt(1. / self.units_list[idx-1])
        elif isinstance(self.wscale, (int, float, np.number)):
            scale = self.wscale
        else:
            raise TypeError("initial weight scale must be float or int type")


        args = [self.units_list[idx-1], self.units_list[idx]]
        self.params['W' + str(idx)] = scale * np.random.randn(*args)
        self.params['b' + str(idx)] = np.zeros(args[-1])


    def _check_layer(self, layer):

        layers = [Linear_Layer, Sigmoid_Layer, Tanh_Layer, Relu_Layer,
                  Softsign_Layer, Softplus_Layer, Elu_Layer, Swish_Layer,
                  Dropout_Layer, Dense, Activation]

        is_layer = False
        for val in layers:
            if isinstance(layer, val):
                is_layer = True

        if is_layer is not True:
            raise ValueError('Could not interpret layer: ' + str(layer))




    def add(self, layer):

        self._check_layer(layer)

        if isinstance(layer, Dense) and self.n_hidden == 0:
            assert layer.input_dim is not None,\
            'Set the units dimension in input layer'
            self.units_list.append(layer.input_dim)

        if isinstance(layer, Dense):
            self.units_list.append(layer.units)
            self.init_W()
            arg = [self.params['W' + str(self.n_hidden)], self.params['b' + str(self.n_hidden)]]
            layer.set_param(*arg)

        self.layers.append(layer)



    def set_loss(self, name:str='sum_squared_error'):
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


    def optimizer(self, method):
        opt_dict = {'sgd':SGD(),'rmsprop':RMSprop(),'momentum':Momentum(),
                    'adam':Adam(),'adagrad':Adagrad(),'adadelta':Adadelta()}
        optim_comp = False

        if isinstance(method, str):
            for key in opt_dict.keys():
                if key == method:
                    self.optim = opt_dict[method]
                    optim_comp = True
            if optim_comp is not True:
                raise KeyError("Not exist optimizer name : {}".format(method))

        for val in opt_dict.values():
            if isinstance(method, type(val)):
                self.optim = method
                optim_comp = True

        if optim_comp is not True:
            raise ValueError('Could not interpret optimizer: ' + str(method))

        return optim_comp


    def compile(self,loss:str, optimizer):

        opt_compiled = self.optimizer(optimizer)
        loss_compiled = self.set_loss(loss)
        is_compiled = opt_compiled and loss_compiled
        return is_compiled


    def predict(self, X):
        x_out = self.feed_forward(X, train_flg=False)
        return self.cost_function.activate(x_out)


    def feed_forward(self, x, train_flg:bool):

        for layer in self.layers:
            if isinstance(layer, Dropout_Layer):
                x = layer.forward(x, is_training=train_flg)
            else:
                x = layer.forward(x)

        return x


    def accuracy(self, x, t):
        y = self.predict(x)
        return self.metric.accuracy(y,t)


    def loss(self, x, t):

        y = self.feed_forward(x, train_flg=True)

        weight_decay = 0
        for idx in range(1, self.n_hidden):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.alpha * np.sum(np.square(W))

        return self.cost_function(y, t) + weight_decay


    def gradient(self, x, t):

        # forward
        self.loss(x, t)

        # backward
        dout = self.cost_function.delta()

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        grads = {}
        affine_idx = np.where([type(obj) is Dense for obj in self.layers])[0]
        for n, idx in enumerate(list(affine_idx)):
            grads['W' + str(n+1)] = self.layers[idx].dW + self.alpha * self.layers[idx].W
            grads['b' + str(n+1)] = self.layers[idx].db

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
