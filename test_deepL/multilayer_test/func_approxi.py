import numpy as np
import matplotlib.pyplot as plt
from deepL_module.nn.multi_layer_nn import Neural_net
from deepL_module.nn.optimizers import *
from deepL_module.base import *

N = 50 # sample
max_iter = 1000 # iteration


heaviside = lambda x : 0.5 * (1 + np.sign(x))
sinusoidal = lambda x : 0.5 + 0.5 * np.sin(np.pi * x)




# training data
train_x = np.linspace(-1, 1, N).reshape(N,1)
multi_func = [np.square, sinusoidal, np.abs, heaviside]
func_name = ['square', 'sinusoidal', 'abs', 'heaviside']
train_acc = {}
for name in func_name:
    train_acc[name] = []

# test data
x = np.arange(-1,1,0.01).reshape(-1,1)

fig = plt.figure(figsize=(15, 9))


for n,func in enumerate(multi_func,1):
    train_y = func(train_x)
    # plotting training data
    ax = fig.add_subplot(2,2,n)
    ax.scatter(train_x, train_y, s=12, color='blue')
    # constructing NN
    model = Neural_net(n_input=1, n_hidden=[10,10,10], n_output=1)
    model.add(['tanh', 'tanh', 'sigmoid', 'linear'])
    model.set_loss('sum_squared_error')
    optimizer = Adam(lr=0.1, beta_1=0.9, beta_2=0.95)
    #-----learning------
    for _ in range(int(max_iter)):
        grads = model.gradient(train_x,train_y)
        optimizer.update(model.params, grads)
        score = model.accuracy(train_x, train_y)
        train_acc[func_name[n-1]].append(score)

    # prediction data
    y = model(x)
    ax.plot(x, y, 'r-')
    plt.xticks([-1,0,1])
    plt.yticks([0,0.5,1])
    plt.ylim([-0.1, 1.1])
    plt.subplots_adjust(wspace=0.2,hspace=0.3)
plt.show()


fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
x = np.arange(max_iter)
for _, name in enumerate(func_name):
    ax.plot(x, train_acc[name], label = name)
plt.legend(fontsize=15)
plt.xlim(-1,max_iter / 10)
plt.show()

'''# save model'''
path_w = './../../prml/deepL_module/datasets/model_data/test_model.pkl'
save_model(model)
