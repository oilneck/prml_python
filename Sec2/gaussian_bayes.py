import numpy as np
import matplotlib.pyplot as plt
from pd import *

fig = plt.figure(figsize = (7,3))
ax = fig.add_subplot(1,2,1)
plt.title('Bayesian inference')

'''Bayesian inference'''
x = np.linspace(-1,1,1000)
model = Gaussian(mu=0.,var=0.1)
ax.plot(x,model.pdf(x),color='k',label="N=0")

color_list=['lime','blue','red','orange','magenta']
for i,n_sample in enumerate([1,2,10]):
    model.mu_fit(np.random.normal(loc=0.8,scale=0.1,size=n_sample))
    ax.plot(x,model.pdf(x),color=color_list[i],label="N={0}".format(n_sample))
plt.xlim(-1,1)
plt.ylim(0,5)
plt.xticks([-1,0,1])
plt.yticks([0,5])
plt.legend(fontsize=12)


'''Maximum likelihood'''
ax = fig.add_subplot(1,2,2)
plt.title('Maximum likelihood')
ax.plot(x,Gaussian(mu = 0.3, var = 0.1**2).pdf(x),label='population',color='k',zorder=3,linestyle='dashed')

for i,n_sample in enumerate([2,1000]):
    model.fit(np.random.normal(loc=0.3,scale=0.1,size=n_sample))
    ax.plot(x,model.pdf(x),color=color_list[i+3],label="N={0}".format(n_sample))
plt.xlim(-1,1)
plt.xticks([-1,0,1])
plt.yticks([0])
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
