import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from deepL_module.base.functions import *
from deepL_module.nn.optimizers import *
from collections import OrderedDict




init_pos = [0.6,1.7]

init_pos = np.array(init_pos).astype(float)

params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0


optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.05)
optimizers["Momentum"] = Momentum(lr=0.02,momentum=0.8)
optimizers["RMSprop"] = RMSprop(lr=0.2,rho=0.95)
optimizers["Adam"] = Adam(lr=0.2,beta_1=0.97,beta_2=0.99)

idx = 1
fig = plt.figure(figsize=(20,10))
for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]

    ax = fig.add_subplot(2, 2, idx)
    train_x, train_y = np.linspace(-4.5,4.5,100), np.linspace(-4.5,4.5,100)
    X, Y = np.meshgrid(train_x,train_y)
    Z = beal_function(X, Y)
    ax.contour(X, Y, Z, levels=np.logspace(0,5,20), norm=LogNorm(), cmap='jet')
    ax.plot(*np.array([3,0.5]),'r*',markersize=18,color='orange')

    for i in range(90):
        x_history.append(params['x'])
        y_history.append(params['y'])

        grads['x'], grads['y'] = grad_beal(params['x'], params['y'])
        optimizer.update(params, grads)

    idx += 1
    path = np.array([x_history,y_history])
    path = path.T[::2].T
    ax.plot(x_history, y_history, 'o-', color="r",markersize=5,markevery=7)
    plt.ylim(-4.5, 4.5)
    plt.xlim(-1.5, 4.)
    plt.title(key,fontsize=20)
    plt.tick_params(labelbottom=False,labelleft=False)
    plt.tick_params(bottom=False,left=False)

plt.show()
