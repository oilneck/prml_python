import numpy as np
from scipy.stats import norm
from wave_function_class import WaveFunction
import matplotlib.pyplot as plt
from scipy import integrate

x = np.linspace(-5, 5, 100)

# simple neural network for finding the wave func.
model = WaveFunction()
model.update(n_iter=50)

# plot wave func.
fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111)
ax.plot(x, model.phi(x) / max(model.phi(x)), c='limegreen', label='Predict')
ax.plot(x, norm.pdf(x) / max(norm.pdf(x)), c='r', label='Ground state')
plt.xlabel('x', fontsize=15)
plt.legend(fontsize=15)
plt.show()
