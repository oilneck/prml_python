import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from deepL_module.nn.optimizers import *


def f(x, y):
    return x ** 2 / 20. + y ** 2


def gradf(x, y):
    return x / 10., 2. * y


X, Y = np.meshgrid(np.linspace(-10,10,100), np.linspace(-5,5,100))
Z = f(X, Y)

init_pos = (-7.0, 3.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0


optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["RMSprop"] = RMSprop(lr=0.15)
optimizers["Adam"] = Adam(lr=0.3)

idx = 1
fig = plt.figure(figsize=(20,10))
for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]
    ax = fig.add_subplot(2,2,idx)


    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])

        grads['x'], grads['y'] = gradf(params['x'], params['y'])
        optimizer.update(params, grads)

    idx += 1
    ax.plot(x_history, y_history, 'o-', color="red",markersize=5,markevery=1)
    ax.contour(X, Y, Z , cmap='jet')
    ax.plot(*np.array([0.,0.]),'r*',markersize=18,color='k')
    plt.ylim(-6, 6)
    plt.xlim(-10, 10)


    plt.title(key,fontsize=20)
    plt.tick_params(labelbottom=False,labelleft=False)
    plt.tick_params(bottom=False,left=False)

plt.show()
