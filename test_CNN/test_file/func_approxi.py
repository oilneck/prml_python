import numpy as np
import matplotlib.pyplot as plt
from deepL_module.nn.two_layer_net import Two_layer_net
from deepL_module.nn.optimizers import *

N = 50 # sample
max_iter = 5000


def multi_func(x,f_name='square'):
    if f_name == 'square':
        f = x**2
    elif f_name == 'heaviside':
        f = 0.5 * (np.sign(x) + 1)
    elif f_name == 'sinusoidal':
        f = 0.5 + 0.5 * np.sin(x * np.pi)
    elif f_name == 'absolute':
        f = np.abs(x)
    return f

# training data
train_x = np.linspace(-1, 1, N).reshape(N,1)
train_y = multi_func(train_x,'heaviside')

# constructing NN
model = Two_layer_net(1,4,1)
model.add(['tanh','linear'])
optimizer = Adam(lr = 0.1)
#---learning----
for _ in range(int(max_iter)):
    grads = model.gradient(train_x,train_y)
    optimizer.update(model.params, grads)


# test data
test_x = np.arange(-1,1,0.01).reshape(-1,1)
test_y = model.predict(test_x)

# Plotting training data
plt.scatter(train_x, train_y,s=10,color='blue')

# Plotting output data in NN model
plt.plot(test_x, test_y, 'r-')
plt.ylim([-0.1, 1.1])
plt.show()
