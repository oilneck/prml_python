from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras import losses
from tensorflow.keras.optimizers import *
import numpy as np
import matplotlib.pyplot as plt

def create_noise_data(sample:int=1000):
    x = np.random.uniform(-1., 1., size=(sample, 2))
    labels = (np.prod(x, axis=1) > 0).astype(np.float)
    return x, labels.reshape(-1, 1)

# training dataset
train_x, train_t = create_noise_data()

# test dataset
X,Y = np.meshgrid(np.linspace(-1, 1, 100),np.linspace(-1, 1, 100))
test_x = np.array([X.ravel(), Y.ravel()]).reshape(2,-1).T


'''Neural network design'''
model = Sequential()
model.add(Dense(4,input_dim=2,activation='tanh'))
model.add(Dense(1,activation='sigmoid'))
optimize_routine = Adam(lr=0.1)
model.compile(optimizer=optimize_routine,loss=losses.mean_squared_error)

# create prediction data
model.fit(train_x,train_t,epochs=100,verbose=1,batch_size=len(train_x))
Z = model.predict(test_x)





# plot the training data
colors = ["blue", "red"]
markers = [".","x"]
set_color = [colors[int(cls_n)] for cls_n in train_t]
set_marker = [markers[int(cls_n)] for cls_n in train_t]
for n in range(len(train_t)):
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
