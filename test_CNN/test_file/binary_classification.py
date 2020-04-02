import numpy as np
import matplotlib.pyplot as plt
from deepL_module.nn.two_layer_net import Two_layer_net
from deepL_module.nn.optimizers import *

def create_noise_data(sample:int=1000):
    x = np.random.uniform(-1., 1., size=(sample, 2))
    labels = (np.prod(x, axis=1) > 0).astype(np.float)
    return x, labels.reshape(-1, 1)

# training dataset
train_x, train_y = create_noise_data()

# test dataset
X,Y = np.meshgrid(np.linspace(-1, 1, 100),np.linspace(-1, 1, 100))
test_x = np.array([X.ravel(), Y.ravel()]).reshape(2,-1).T

# constructing NN
model = Two_layer_net(2,3,1)
optimizer = Adam(lr = 0.01)
#---learning----
for _ in range(int(1000)):
    grads = model.gradient(train_x,train_y)
    optimizer.update(model.params, grads)


# plot the training data
colors = ["blue", "red"]
markers = [".","x"]
set_color = [colors[int(cls_n)] for cls_n in labels]
set_marker = [markers[int(cls_n)] for cls_n in labels]
for n in range(len(labels)):
    plt.scatter(train_x[n, 0], train_x[n, 1], c=set_color[n],marker=set_marker[n],s=15)

# plot the test data
plt.contourf(X, Y, Z.reshape(X.shape), levels=np.linspace(0, 1, 11), alpha=0.2,cmap='jet')
plt.colorbar()
# axis setting
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.show()
