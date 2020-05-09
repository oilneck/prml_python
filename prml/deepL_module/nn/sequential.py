import numpy as np
from deepL_module.nn.layers import *
from deepL_module.nn.cost_functions import *
from deepL_module.nn.optimizers import *
from deepL_module.nn.metrics import *
from deepL_module.base import *
from collections import OrderedDict


class Sequential(object):

    def __init__(self, w_std:float=None, alpha:float=0.):
        self.alpha = alpha # Weight decay coefficient.
        self.wscale = w_std
        self.batch_num = 0
        self.idx = 0
        self.units_num = []
        self.conv_params = []
        self.params = {}

        self.layers = []

        self.cost_function = None
        self.metric = None
        self.optim = None



    def __call__(self,X):
        return self.predict(X)

    def init_W(self):

        self.idx += 1

        scale = 1.
        if self.wscale is None:
            scale = np.sqrt(1. / self.units_num[-1])
        elif isinstance(self.wscale, (int, float, np.number)):
            scale = self.wscale
        else:
            raise TypeError("initial weight scale must be float or int type")


        args = [self.units_num[-2], self.units_num[-1]]
        self.params['W' + str(self.idx)] = scale * np.random.randn(*args)
        self.params['b' + str(self.idx)] = np.zeros(args[-1])

    def init_conv(self, filter_num, filter_size, input_shape, stride, pad):

        self.idx += 1
        args = [filter_num, self.conv_params[-2][0], filter_size[0], filter_size[1]]
        self.params['W' + str(self.idx)] = 0.01 * np.random.randn(*args)
        self.params['b' + str(self.idx)] = np.zeros(filter_num)

    def init_batch(self):

        if self.units_num == []:
            raise Exception("Could not set 'batch norm' before 'dense' one")

        self.batch_num += 1

        _shape = self.units_num[-1]
        self.params['gamma' + str(self.batch_num)] = np.ones(_shape)
        self.params['beta' + str(self.batch_num)] = np.zeros(_shape)


    def _check_layer(self, layer):

        layers = [Linear_Layer, Sigmoid_Layer, Tanh_Layer, Relu_Layer,
                  Softsign_Layer, Softplus_Layer, Elu_Layer, Swish_Layer,
                  Dropout_Layer, Batch_norm_Layer, Dense, Activation,
                  Maxpooling, Conv2D]

        is_layer = False
        for val in layers:
            if isinstance(layer, val):
                is_layer = True

        if is_layer is not True:
            raise ValueError('Could not interpret layer: ' + str(layer))


    def add(self, layer):

        self._check_layer(layer)

        if isinstance(layer, Dense) and self.units_num == []:
            assert layer.input_dim is not None,\
            'Set the the number of units in input layer'
            self.units_num.append(layer.input_dim)

        if isinstance(layer, Dense):
            self.units_num.append(layer.units)
            self.init_W()
            args = [self.params['W' + str(self.idx)], self.params['b' + str(self.idx)]]
            layer.set_param(*args)

        # convolution init
        if isinstance(layer, Conv2D) and self.conv_params == []:
            self.conv_params.append(layer.input_shape)


        if isinstance(layer, Conv2D):
            input_size = self.conv_params[-1][1]
            output_size = calc_size(input_size, layer.kernel_size[0], layer.stride, layer.pad)
            self.conv_params.append((layer.filters,int(output_size),int(output_size)))
            self.init_conv(*layer.cache)
            args = [self.params['W' + str(self.idx)], self.params['b' + str(self.idx)]]
            layer.set_param(*args)

        if isinstance(layer, Maxpooling):
            _shape = self.conv_params[-1]
            output_shape = (_shape[0], int(_shape[1] / 2), int(_shape[2] / 2))
            self.conv_params.append(output_shape)
            self.units_num.append(np.prod(output_shape))

        if isinstance(layer, Batch_norm_Layer):
            self.init_batch()
            args = [self.params['gamma' + str(self.batch_num)],
                    self.params['beta' + str(self.batch_num)]]
            layer.set_param(*args)

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
            if isinstance(layer, (Dropout_Layer, Batch_norm_Layer)):
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
        num_W = np.sum(['W' in key for key in list(self.params.keys())])
        for idx in range(1, num_W):
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
        W_loc = np.where([isinstance(obj,(Dense, Conv2D)) for obj in list(self.layers)])[0]
        for n, idx in enumerate(list(W_loc)):
            grads['W' + str(n+1)] = self.layers[idx].dW + self.alpha * self.layers[idx].W
            grads['b' + str(n+1)] = self.layers[idx].db

        batch_loc = np.where([type(obj) is Batch_norm_Layer for obj in self.layers])[0]
        for n, idx in enumerate(list(batch_loc)):
            grads['gamma' + str(n+1)] = self.layers[idx].dgamma
            grads['beta' + str(n+1)] = self.layers[idx].dbeta

        return grads


    def fit(self, X_train:np.ndarray, t_train:np.ndarray, epochs:int=1, batch_size:int=None, history:bool=False):

        hist = {}
        hist['loss'] = []
        hist['acc'] = []

        train_size = X_train.shape[0]
        iter_per_epoch = 1

        if batch_size is not None:
            iter_per_epoch = int(max(train_size / batch_size, 1))

        n_iter = epochs * iter_per_epoch

        for n in range(int(n_iter)):
            x_batch, t_batch = get_mini_batch(X_train, t_train, batch_size)
            grads = self.gradient(x_batch, t_batch)
            self.optim.update(self.params, grads)

            if history:
                if n % iter_per_epoch == 0:
                    loss = self.loss(x_batch, t_batch)
                    score = self.accuracy(x_batch, t_batch)
                    hist['loss'].append(loss)
                    hist['acc'].append(score)
            else:
                hist = None


        self.history = hist
        return hist

    def save(self, path:str=None, name:str=None):
        save_model(self, path, name)
