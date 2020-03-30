import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)

x = np.linspace(-6,6,100)
y = relu(x)
plt.plot(x,y,color='r')
plt.xlim(-6,6)
plt.xticks(np.arange(-6,6.1,2))
plt.show()
