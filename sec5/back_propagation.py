#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

N = 50 # The num. of test data
ETA = 0.1 # Learning rate
MAX_LOOP = 10000 # loops

NUM_INPUT = 2 # number of input unit (including bias parameter)
NUM_HIDDEN = 4 #  number of hidden unit (including bias param.)
NUM_OUTPUT = 1 # number of output unit



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

def Forward_propagation(x):
    input_vec_x = np.insert([x],0,1)
    z = np.tanh(w1 @ input_vec_x)
    y = w2 @ z
    return y, z

def Back_propagation(x,t):
    y,z = Forward_propagation(x)
    delta_2 = y - t
    delta_1 = (1 - z * z) * (w2.T @ delta_2)
    return delta_1,delta_2



# training data
xlist = np.linspace(-1, 1, N).reshape(N, 1)
tlist = multi_func(xlist,'heaviside')
w1 = np.random.random((NUM_HIDDEN,NUM_INPUT))
w2 = np.random.random((NUM_OUTPUT,NUM_HIDDEN))



for loop in range(MAX_LOOP):
    for n in range(len(xlist)):
        y,z = Forward_propagation(xlist[n])
        del_1,del_2 = Back_propagation(xlist[n],tlist[n,:])
        w1 -= ETA * (del_1.reshape(NUM_HIDDEN,1) @ np.insert(xlist[n],0,1).reshape(1,NUM_INPUT))
        w2 -= ETA * (del_2.reshape(NUM_OUTPUT,1) @ z.reshape(1,NUM_HIDDEN))




ylist = np.zeros((N, NUM_OUTPUT))
zlist = np.zeros((N, NUM_HIDDEN))
for n in range(N):
    ylist[n], zlist[n] = Forward_propagation(xlist[n])



# Plotting training data
plt.scatter(xlist, tlist,s=10,color='blue')

# Plotting output data in Neural Network model
plt.plot(xlist, ylist, 'r-')


# Plotting the output of hidden layer
color_list = ['goldenrod','purple','lime']
for i in range(NUM_HIDDEN-1):
    plt.plot(xlist, zlist[:,i+1],color=color_list[i],linestyle='dashed',linewidth=1)
plt.ylim([-0.1, 1.1])
plt.show()
