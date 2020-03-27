import numpy as np
import matplotlib.pyplot as plt
from nn.feedforward_nn import Feed_Forward

N = 50 # The number of test data

def heaviside(x):
    return 0.5 * (np.sign(x) + 1)


# training data
train_x = np.linspace(-1, 1, N).reshape(N, 1)
train_y = heaviside(train_x)

# Construncting NeuralNet
model = Feed_Forward(1,3,1)
model.compile(optimizer='adam')
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
