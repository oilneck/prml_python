import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(x):
    return 0.5 + 0.5 * np.tanh(0.5 * x)

def step_function(x):
    return (x > 0).astype(float)

x = np.linspace(-5,5,100)
y1 = sigmoid_function(x)
y2 = step_function(x)

plt.plot(x,y1,color='r',linestyle='solid')
plt.plot(x,y2,color='b',linestyle='dashed')
plt.show()
