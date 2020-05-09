import numpy as np
import matplotlib.pyplot as plt
from deepL_module.nn.sequential import Sequential
from deepL_module.nn.optimizers import *
from deepL_module.nn.layers import *
from deepL_module.nn.layers.core import *

N = 50 # sample
max_epochs = 1000 # epochs


heaviside = lambda x : 0.5 * (1 + np.sign(x))
sinusoidal = lambda x : 0.5 + 0.5 * np.sin(np.pi * x)




# training data
train_x = np.linspace(-1, 1, N).reshape(N,1)
multi_func = [np.square, sinusoidal, np.abs, heaviside]
func_name = ['square', 'sinusoidal', 'abs', 'heaviside']
train_acc = {}

# test data
x = np.arange(-1,1,0.01).reshape(-1,1)

fig = plt.figure(figsize=(15, 9))


for n,func in enumerate(multi_func,1):
    train_y = func(train_x)
    # plotting training data
    ax = fig.add_subplot(2,2,n)
    ax.scatter(train_x, train_y, s=12, color='blue')
    # constructing NN
    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    routine = Adam(lr=0.1, beta_1=0.9, beta_2=0.95)
    model.compile(loss='sum_squared_error', optimizer=routine)
    #-----learning------
    history = model.fit(train_x, train_y, epochs=max_epochs, history=False)
    #train_acc[func_name[n-1]] = history['acc']

    # prediction data
    y = model(x)
    ax.plot(x, y, 'r-')
    plt.xticks([-1,0,1])
    plt.yticks([0,0.5,1])
    plt.ylim([-0.1, 1.1])
    plt.subplots_adjust(wspace=0.2,hspace=0.3)
plt.show()


# fig = plt.figure(figsize=(10,4))
# ax = fig.add_subplot(111)
# x = np.arange(max_iter)
# for _, name in enumerate(func_name):
#     ax.plot(x, train_acc[name], label = name)
# plt.legend(fontsize=15)
# plt.xlim(-1,max_iter / 10)
# plt.show()
