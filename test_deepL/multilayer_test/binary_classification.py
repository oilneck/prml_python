import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from deepL_module.nn.multi_layer_nn import Neural_net
from deepL_module.nn.optimizers import *

max_iter = 250

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
model = Neural_net(2,[10,10,10],1)
model.add(['tanh', 'swish', 'relu'])
model.set_loss('binary_crossentropy')
optimizer = Adam(lr = 0.1)
score_acc = []

#---learning----
for _ in range(max_iter):
    grads = model.gradient(train_x,labels)
    optimizer.update(model.params, grads)

    score = model.accuracy(train_x,labels)
    score_acc.append(np.asarray(score))


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

# plot the accuracy score
fig = plt.figure(figsize=(7,4))
plt.plot(np.arange(max_iter), score_acc, color='r')
plt.title('Accuracy score', fontsize=15)
plt.xlabel('iteration', fontsize=15)
plt.tight_layout()
plt.show()
