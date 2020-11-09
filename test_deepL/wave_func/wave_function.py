import numpy as np
from scipy.stats import norm
from simple_wave import PredictWaveFunction
import matplotlib.pyplot as plt
from scipy import integrate

x = np.linspace(-5, 5, 100)

# simple neural network for finding the wave func.
model = PredictWaveFunction(2)
# learning params
model.update(n_iter=50)


# plot wave func.
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot(121)
ax.plot(x, model.phi(x) / max(model.phi(x)), c='limegreen', label='Predict')
ax.plot(x, norm.pdf(x) / max(norm.pdf(x)), c='r', label='Ground state')
plt.legend(fontsize=15)

# drawing the error
ax = fig.add_subplot(122)
energies = model.info['energies']
ax.plot(np.arange(len(energies)), energies)
plt.show()
