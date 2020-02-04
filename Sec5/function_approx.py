import numpy as np
import matplotlib.pyplot as plt
from nn.linear_nn import Linear_NeuralNet

N = 50 # The number of test data

def multi_func(x,f_name='square'):
    if f_name == 'square':
        f = x**2
    elif f_name == 'heaviside':
        f = 0.5 * (np.sign(x) + 1)
    elif f_name == 'sinusoidal':
        f = 0.5+0.5 * np.sin(x * np.pi)
    elif f_name == 'absolute':
        f = np.abs(x)
    return f

# training data
train_x = np.linspace(-1, 1, N).reshape(N, 1)
train_y = multi_func(train_x,'sinusoidal')

# Construncting NeuralNet
model = Linear_NeuralNet(2,3,1)
model.fit(train_x,train_y)
Forward_propagation = np.vectorize(model.Forward_propagation)

# test data
test_x = np.arange(-1,1,0.01)
test_y = Forward_propagation(test_x)


# Plotting training data
plt.scatter(train_x, train_y,s=10,color='blue')

# Plotting output data in Neural Network model
plt.plot(test_x, test_y, 'r-')
plt.ylim([-0.1, 1.1])
plt.show()


# Plotting the output of hidden layer
color_list = ['goldenrod','purple','lime']
test_z = model.get_hidden_output(test_x)
for i in range(np.size(test_z,1)):
    plt.plot(test_x, test_z[:,i],color=color_list[i],linestyle='dashed',linewidth=1)
