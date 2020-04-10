import numpy as np

def identity_function(x):
    return x

def tanh(x):
    return np.tanh(x)


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 0.5 + 0.5 * np.tanh(0.5 * x)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    max_val = np.max(x, axis=-1, keepdims=True)
    a = x - max_val
    return np.exp(a) / np.sum(np.exp(a), axis=-1, keepdims=True)


def softsign(x):
    return x / (1. + np.abs(x))


def softplus(x):
    return np.log(1. + np.exp(x))


def elu(x,alpha:float=1.0):
    return np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1) )


def swish(x,beta:float=1.0):
    return x * sigmoid(beta * x)


def beal_function(x,y):
    term_1 = np.square(1.5 - x + x * y)
    term_2 = np.square(2.25 - x + x * y ** 2)
    term_3 = np.square(2.625 - x + x * y ** 3)
    return term_1 + term_2 + term_3


def grad_beal(x,y):
    min,max = 1e-8,1e+10
    x,y = np.clip(x,min,max), np.clip(y,min,max)
    term_1 = 2 * (1.5 - x + x * y)
    term_2 = 2 * (2.25 - x + x * y ** 2)
    term_3 = 2 * (2.625 - x + x * y ** 3)
    del_x = term_1 * (y - 1) + term_2 * (y ** 2 - 1) + term_3 * (y ** 3 - 1)
    del_y = term_1 * x + term_2 * (2 * x * y) + term_3 * (3 * x * y ** 2)
    return del_x, del_y
