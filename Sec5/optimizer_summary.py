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
model.add(['tanh','identity'])

# test data
x = np.arange(-1,1,0.01)

# Plotting training data
color_list = ['blue','orange','lime','red']
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.scatter(train_x, train_y,s=8,color=color_list[0],zorder=2)

# Plotting output data
lr_list = [0.2,0.01,0.2]
for n,(routine,lr) in enumerate(zip(['SGD','RMSprop','Adam'],lr_list),1):
    model.optimizers(method = routine.lower())
    model.fit(train_x,train_y,n_iter=1500,learning_rate=lr)
    y = model(x)
    ax.plot(x,y,color_list[n],zorder=3,label=routine,linewidth=2.3,alpha=0.8)
plt.legend(fontsize=15)
plt.show()
