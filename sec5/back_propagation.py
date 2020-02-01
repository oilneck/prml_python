#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from nn.linear_nn import Linear_NeuralNet
N = 50 # The num. of test data
n_input = 2
n_hidden = 4
n_output = 1

def multi_func(x,f_name='square'):
    if f_name == 'square':
        f = x**2
    elif f_name == 'heaviside':
        f = 0.5 * (np.sign(x) + 1)
    elif f_name == 'sinusoidal':
        f = 0.5+0.5 * np.sin(xlist *np.pi)
    elif f_name == 'absolute':
        f = np.abs(x)
    return f




# training data
xlist = np.linspace(-1, 1, N).reshape(N, 1)
tlist = multi_func(xlist,'heaviside')
model = Linear_NeuralNet()
model.fit(xlist,tlist)
ylist = np.zeros((N, n_output))
zlist = np.zeros((N, n_hidden))
for n in range(N):
    ylist[n],zlist[n] = model.Forward_propagation(xlist[n]),model.layer1.output

# Plotting training data
plt.scatter(xlist, tlist,s=10,color='blue')

# Plotting output data in Neural Network model
plt.plot(xlist, ylist, 'r-')


# Plotting the output of hidden layer
color_list = ['goldenrod','purple','lime']
for i in range(n_hidden-1):
    plt.plot(xlist, zlist[:,i+1],color=color_list[i],linestyle='dashed',linewidth=1)
plt.ylim([-0.1, 1.1])
plt.show()
