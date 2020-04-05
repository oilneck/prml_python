import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return (x > 0).astype(float)

x = np.linspace(-5,5,1000)
y = step_function(x)
plt.plot(x,y,c='r')
plt.ylim(-0.1,1.1)
plt.show()
