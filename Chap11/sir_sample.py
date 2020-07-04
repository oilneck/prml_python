import numpy as np
import matplotlib.pyplot as plt
from pd import *
from sampling import SIR


# target function p(z)
func = lambda x : Gamma(alpha=5, beta=1).pdf(x)
# proposal distribution q(z)
proba = Uniform(0, 20)

sampler = SIR(target=func, prop=proba)


x = np.linspace(0, 20, 200)
fig = plt.figure(figsize=(15,4))
for n, n_sample in enumerate([100,1000,100000]):
    ax = fig.add_subplot(1, 3, n+1)
    xs = sampler.rvs(n_sample)
    ax.plot(x, func(x), c='limegreen', label=r'$\tilde{p}(z)$: gamma', lw=2)
    ax.hist(xs, bins=50, color='aqua', label='SIR samples', density=1, ec='k', alpha=.2)
    ax.set_ylim(0, 0.25)
plt.plot(x, proba.pdf(x), c='r', label=r'$q(z)$: proposal')
plt.legend(bbox_to_anchor=(1.05, 0.), loc='lower left', borderaxespad=0, fontsize=15)
plt.tight_layout()
plt.show()
