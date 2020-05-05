import numpy as np
from deepL_module.nn.layers import *

class Activation():

    def __init__(self, activation):
        self.func = self.__getFunc(activation)

    def __call__(self, inputs):
        return self.func.forward(inputs)

    def get(self):
        return self.func

    def __getFunc(self, obj):

        act_dict = {'linear':Linear_Layer(),
                    'sigmoid':Sigmoid_Layer(),
                    'tanh':Tanh_Layer(),
                    'relu':Relu_Layer(),
                    'elu':Elu_Layer(),
                    'softsign':Softsign_Layer(),
                    'softplus':Softplus_Layer(),
                    'swish':Swish_Layer()
                    }

        activate = None
        is_act = False

        if isinstance(obj, str):
            for key in act_dict.keys():
                if key == obj:
                    activate = act_dict[obj]
                    is_act = True
            if is_act is not True:
                raise KeyError("Not exist funcion name : {}".format(obj))

        for val in act_dict.values():
            if isinstance(obj, type(val)):
                activate = obj
                is_act = True

        if is_act is not True:
            raise ValueError('Could not interpret identifier : ' + str(obj))

        return activate

    def forward(self, x):
        return self.func.forward(x)

    def backward(self, delta):
        return self.func.backward(delta)
