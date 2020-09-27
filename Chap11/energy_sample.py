import numpy as np
import matplotlib.pyplot as plt
from sampling import SASampling

N = 10
h = dict(zip(np.arange(N), np.random.uniform(-1, 1, size=N)))
J = {(i, j): np.random.uniform(-1, 1) for i in range(N) for j in range(i+1, N) if i!=j}


sampler = SASampling()
result = sampler.sample_ising(h, J, n_iter=100)


print('optimum state:', sampler.sample)
plt.hist(sampler.energies, bins=10, density=True, color='b', rwidth=.7, ec='k', lw=1.)
plt.xlabel('Energy', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()
