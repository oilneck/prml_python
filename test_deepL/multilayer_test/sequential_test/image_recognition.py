import numpy as np
import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from deepL_module.nn.sequential import Sequential
from deepL_module.nn.optimizers import *
from deepL_module.base import *
from deepL_module.nn.layers import *


max_epochs = 50


'''#0 loading data '''
(X_train, train_t), (X_test, test_t) = load_mnist(normalize=True, flatten=False)
X_train, train_t = X_train[:10000], train_t[:10000]
train_t = to_categorical(train_t)


'''#1 config for NN '''
model = Sequential()
model.add(Conv2D(16,(5,5),input_shape=(1,28,28)))
model.add(Activation('relu'))
model.add(Maxpooling(pool_h=2, pool_w=2, stride=2))
model.add(Conv2D(16,(3,3)))
model.add(Activation('relu'))
model.add(Maxpooling(pool_h=2, pool_w=2, stride=2))
model.add(Dense(100, activation='relu'))
model.add(Dense(10))
# optimizer
routine = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=routine)


'''#2 learning '''
import time
start = time.time()
hist = model.fit(X_train, train_t,
                 batch_size=256,
                 epochs=max_epochs,
                 history=True)
print(time.time()-start)



'''#3 drawing loss & accuracy data '''
# ----- loss data -----
fig = plt.figure(figsize=(15,4)) # dpi=50
ax1 = fig.add_subplot(121)
x = np.arange(max_epochs)
ax1.plot(x, hist['loss'], color='blue',marker="*", markersize=7, markevery=10, label='loss')
ax1.set_xlabel("epochs",fontsize=25)
ax1.set_title("loss",fontsize=25)
ax1.set_xlim([-2,max_epochs])
plt.show()

# --- accuracy data ---
ax = fig.add_subplot(122)
ax.plot(x, smooth_filt(hist['acc']),
        color = 'r',
        marker = "*",
        markersize = 8,
        markevery = 10)
plt.xlabel("epochs",fontsize=25)
plt.title("accuracy",fontsize=25)
plt.xlim([-2,50])
plt.tight_layout()
plt.show()

#model.save(name='simple_CNN')
