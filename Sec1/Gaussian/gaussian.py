import numpy as np
from pd import *
import matplotlib.pyplot as plt

# ave
mu = 0

# std
sigma = np.sqrt(10)

# confidence interval
conf_L = mu - sigma
conf_R = mu + sigma
fill_x = np.arange(conf_L, conf_R, 0.01)

'''Gaussian distribution'''
prob = Gauss(mu, sigma ** 2)
x = np.linspace(conf_L * 4, conf_R * 4, 1000)
y = prob.pdf(x)


# Plotting
fig = plt.figure(figsize=(6.2,4))
ax = fig.add_subplot(111)
ax.plot(x, y, color='red')
plt.text(conf_L * 3.5, 0.8 * max(y), r"$\mathcal{N}(x|\mu,\sigma^2)$",fontsize=15)
plt.fill_between(x, y, where = x > mu - sigma, color='b', alpha=0.2)
plt.fill_between(x, y, where = x > mu + sigma, color='w', alpha=1)
plt.vlines(x=mu, ymin=0, ymax=prob.pdf(mu), linewidth=1,linestyle='--',color = 'k')
plt.vlines(x=conf_L, ymin=0, ymax=prob.pdf(conf_L), linewidth=1,linestyle='--',color = 'k')
plt.vlines(x=conf_R, ymin=0, ymax=prob.pdf(conf_R), linewidth=1,linestyle='--',color = 'k')
plt.xticks([conf_L * 4, conf_L, mu, conf_R, conf_R * 4],("",r"$\mu-\sigma$",r"$\mu$",r"$\mu+\sigma$","x"),fontsize=15)
plt.yticks([0])
plt.xlim(conf_L * 4, conf_R * 4)
plt.ylim(0,max(y) + 0.01)
plt.tight_layout()
plt.show()
