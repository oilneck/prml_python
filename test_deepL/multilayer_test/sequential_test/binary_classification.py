import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from deepL_module.nn.optimizers import *
from deepL_module.nn.sequential import Sequential
from deepL_module.nn.layers import *

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
model = Sequential(2)
model.add(Dense(32))
model.add(Tanh_Layer())
model.add(Dense(10))
model.add(Dropout_Layer(0.15))
model.add(Swish_Layer())
model.add(Dense(1))
routine = Adam(lr = 0.1)
model.compile(loss='binary_crossentropy', optimizer=routine)

#---learning----
history = model.fit(train_x, labels, n_iter=max_iter, history=True)



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
plt.plot(np.arange(max_iter), history['acc'], color='r')
plt.title('Accuracy score', fontsize=15)
plt.xlabel('iteration', fontsize=15)
plt.tight_layout()
plt.show()
