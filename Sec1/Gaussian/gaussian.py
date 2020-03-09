import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
F_size = 15

# ave
mu = 1
# std
sigma = 10

# left-margin
start = mu - sigma * 3

# right-margin
end = mu + sigma * 3

X = np.arange(start, end, 0.001)
fill_X = np.arange(mu-sigma,mu+sigma,0.01)
Y = norm.pdf(X, loc=mu, scale=sigma)
# Plotting
plt.yticks( [0,10] )
#plt.xticks( [mu] )
plt.xticks([mu-sigma,mu,mu+sigma],(r"$\mu-\sigma$",r"$\mu$",r"$\mu+\sigma$"),fontsize=F_size)
plt.text(end,-0.005,r"$x$",fontsize=F_size)
plt.text(mu-30,0.035,r"$\mathcal{N}(x|\mu,\sigma^2)$",fontsize=F_size)
plt.fill_between(X,Y,where=X>mu-sigma, color='b', alpha=0.2)
plt.fill_between(X,Y,where=X>mu+sigma, color='w', alpha=1)
plt.axvline(x=mu, ymin=0, ymax=0.94, linewidth=1,linestyle='--',color = 'k')
plt.axvline(x=mu-sigma, ymin=0, ymax=0.6, linewidth=1,linestyle='--',color = 'k')
plt.axvline(x=mu+sigma, ymin=0, ymax=0.6, linewidth=1,linestyle='--',color = 'k')
plt.plot(X, Y, color='red')
plt.show()