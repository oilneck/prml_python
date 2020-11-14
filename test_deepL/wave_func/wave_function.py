import numpy as np
from scipy.stats import norm
from simple_wave import PredictWaveFunction
import matplotlib.pyplot as plt
from scipy import integrate

x = np.linspace(-5, 5, 100)

# simple neural network for finding the wave func.
model = PredictWaveFunction(2)
# learning params
#model.update(n_iter=50)
model.load()


# plot wave func.
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(121)
ax.plot(x, model.phi(x) / max(model.phi(x)), c='limegreen', label='Predict')
ax.plot(x, norm.pdf(x) / max(norm.pdf(x)), c='r', label='Ground state')
plt.xlabel('x', fontsize=15)
plt.legend(fontsize=15)

# drawing the error
ax = fig.add_subplot(122)
energies = model.info['energies']
ax.plot(np.arange(len(energies)), energies)
plt.title('Energies', fontsize=15)
plt.xlabel('epochs', fontsize=15)
plt.tight_layout()
plt.show()
