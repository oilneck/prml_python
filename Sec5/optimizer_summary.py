import numpy as np
import matplotlib.pyplot as plt
from nn.adam import Adam
from nn.linear_nn import Linear_NeuralNet

N = 50 # The number of test data

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
train_x = np.linspace(-1, 1, N).reshape(N, 1)
train_y = multi_func(train_x,'heaviside')

# Construncting NeuralNet
model = Linear_NeuralNet(1,3,1)
model = Adam(1,3,1)
model.fit(train_x,train_y,learning_rate=0.3,n_iter=1500)

# test data
test_x = np.arange(-1,1,0.01)
test_y = model(test_x)


# Plotting training data
plt.scatter(train_x, train_y,s=10,color='blue')

# Plotting output data in NN model
plt.plot(test_x, test_y, 'r-')
plt.ylim([-0.1, 1.1])
plt.show()
