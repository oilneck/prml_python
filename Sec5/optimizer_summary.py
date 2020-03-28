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
color_list = ['blue','orange','black','lime','skyblue','red']
style_list = ['solid','solid','dashdot','dashed','solid','solid']
fig = plt.figure(figsize=(11,6.5))
ax = fig.add_subplot(111)
ax.scatter(train_x, train_y,s=25,color=color_list[0],linestyle=style_list[0],zorder=2)

# Plotting output data
lr_list = [0.2,0.2,0.02,0.3,0.2]
for n,(routine,lr) in enumerate(zip(['SGD','Momentum','RMSprop','Adagrad','Adam'],lr_list),1):
    model.optimizer(method = routine.lower())
    model.fit(train_x,train_y,n_iter=1500,learning_rate=lr)
    y = model(x)
    ax.plot(x,y,color_list[n],
            zorder=3,label=routine,
            linestyle=style_list[n],
            linewidth=2.3,alpha=0.8)
    model.clear()
plt.legend(fontsize=15)
plt.xlim(-1,1)
plt.xticks([-1,0,1])
plt.show()
