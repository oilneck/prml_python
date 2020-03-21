from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras import losses
from tensorflow.keras.optimizers import *
import numpy as np
import matplotlib.pyplot as plt

N = 50

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

# create training data
train_x = np.linspace(-1,1,N)
train_y = multi_func(train_x,'heaviside')


'''Neural network design'''
# declaration
model = Sequential()

# setting for hidden layer & output layer
model.add(Dense(3,input_dim=1,activation='tanh'))
model.add(Dense(1,activation='linear'))

# setting optimizer (lr:learning rate)
optimize_routine = Adam(lr=0.15)

# selection of loss function
model.compile(optimizer=optimize_routine,loss=losses.mean_squared_error)


'''creating prediction data'''
model.fit(train_x,train_y,epochs=1000,verbose=0)
test_x = np.linspace(-1,1,1000)
test_y = model.predict(test_x)

# output in hidden layer
middle_model = Model(inputs=model.input, outputs=model.layers[0].output)
middle_output = middle_model.predict(test_x)

# plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(train_x, train_y,s=10,color='blue')
ax.plot(test_x,test_y,color='red')
ax.plot(test_x,middle_output,linestyle='dashed',linewidth=1)
plt.ylim([-0.15, 1.15])
plt.show()
