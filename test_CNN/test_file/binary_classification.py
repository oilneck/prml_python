import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from deepL_module.nn.two_layer_net import Two_layer_net
from deepL_module.nn.optimizers import *

def create_noise_data(sample:int=400):
    x,labels = make_moons(sample, noise = 0.2)
    x[:,0] -= 0.5
    return x, labels.reshape(-1, 1)

# training dataset
train_x, labels = create_noise_data()

# test dataset
X,Y = np.meshgrid(np.linspace(-3, 3, 100),np.linspace(-3, 3, 100))
test_x = np.array([X.ravel(), Y.ravel()]).reshape(2,-1).T

# constructing NN
model = Two_layer_net(2,3,1)
model.add(['tanh','sigmoid'])
optimizer = Adam(lr = 0.1)
#---learning----
for _ in range(int(250)):
    grads = model.gradient(train_x,labels)
    optimizer.update(model.params, grads)


# plot the training data
plt.scatter(train_x[labels.ravel() == 0,0],
            train_x[labels.ravel() == 0,1],
            marker = ".", s = 20, color = 'b')

plt.scatter(train_x[labels.ravel() == 1,0],
            train_x[labels.ravel() == 1,1],
            marker = "x", s = 20, color = 'r')


# plot the test data
Z = model.predict(test_x)
plt.contourf(X, Y, Z.reshape(X.shape), levels=np.linspace(0, 1, 11), alpha=0.2,cmap='jet')
plt.colorbar()
# axis setting
plt.xlim(-2.2,2.2)
plt.ylim(-2.,2.)
plt.xticks(np.arange(-2,3,1))
plt.yticks(np.arange(-2,3,1))
plt.show()
