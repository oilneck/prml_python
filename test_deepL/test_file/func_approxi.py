import numpy as np
import matplotlib.pyplot as plt
from deepL_module.nn.two_layer_net import Two_layer_net
from deepL_module.nn.optimizers import *

N = 50 # sample
max_iter = 2000 # iteration


heaviside = lambda x : 0.5 * (1 + np.sign(x))
sinusoidal = lambda x : 0.5 + 0.5 * np.sin(np.pi * x)




# training data
train_x = np.linspace(-1, 1, N).reshape(N,1)
multi_func = [np.square, sinusoidal, np.abs, heaviside]

# test data
x = np.arange(-1,1,0.01).reshape(-1,1)

fig = plt.figure(figsize=(15, 9))


for n,func in enumerate(multi_func,1):
    train_y = func(train_x)
    # plotting training data
    ax = fig.add_subplot(2,2,n)
    ax.scatter(train_x, train_y,s=12,color='blue')
    # constructing NN
    model = Two_layer_net(1,4,1)
    model.add(['tanh','linear'])
    optimizer = Adam(lr = 0.1, beta_1 = 0.95, beta_2 = 0.95)
    #-----learning------
    for _ in range(int(max_iter)):
        grads = model.gradient(train_x,train_y)
        optimizer.update(model.params, grads)
    # prediction data
    y = model.predict(x)
    ax.plot(x, y, 'r-')
    plt.xticks([-1,0,1])
    plt.yticks([0,0.5,1])
    plt.ylim([-0.1, 1.1])
    plt.subplots_adjust(wspace=0.2,hspace=0.3)
plt.show()
