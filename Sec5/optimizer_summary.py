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
color_list = ['blue','orange','red']
plt.scatter(train_x, train_y,s=10,color=color_list[0],zorder=3)

# Plotting output data
for n,routine in enumerate(['sgd','adam'],1):
    model.compile(optimizer = routine)
    model.fit(train_x,train_y,n_iter=2000,learning_rate=0.2)
    y = model(x)
    plt.plot(x,y,color_list[n],zorder=1,label=routine,linewidth=2)
plt.legend(fontsize=15)
plt.show()
