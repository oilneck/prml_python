import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(x):
    return 0.5 + 0.5 * np.tanh(0.5 * x)

x = np.linspace(-5,5,1000)
y = sigmoid_function(x)
plt.plot(x, y, color = 'r')
plt.ylim(-0.1,1.1)
plt.show()
