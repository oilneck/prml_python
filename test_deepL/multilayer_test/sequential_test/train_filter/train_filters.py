import numpy as np
import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from deepL_module.nn.sequential import Sequential
from deepL_module.nn.optimizers import *
from deepL_module.base import *
from deepL_module.nn.layers import *


max_epochs = 20


'''#0 loading data '''
(X_train, train_t), (X_test, test_t) = load_mnist(flatten=False)
train_t = to_categorical(train_t)


'''#1 config for NN '''
model = Sequential(w_std=0.01)
model.add(Conv2D(30,(5,5),input_shape=(1,28,28)))
model.add(Activation('relu'))
model.add(Maxpooling(pool_h=2, pool_w=2, stride=2))
model.add(Dense(100, activation='relu'))
model.add(Dense(10))
# optimizer
routine = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=routine)


'''#2 learning '''
import time
start = time.time()
hist = model.fit(X_train, train_t,
                 batch_size=100,
                 epochs=max_epochs,
                 history=True)
print(time.time()-start)

model.save(name='visualize_filter_CNN')
