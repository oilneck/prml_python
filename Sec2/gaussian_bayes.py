import numpy as np
import matplotlib.pyplot as plt
from pd import *

fig = plt.figure(figsize = (12,3.5))
ax = fig.add_subplot(1,3,1)
plt.title('Bayesian inference about ' + r"$\mu$")

'''Bayesian inference about mean'''
x = np.linspace(-1,1,1000)
model = Gaussian(mu=0.,var=0.1)
model.set_param(ave = model , var=0.1)
ax.plot(x,model.pdf(x),color='k',label="N=0")

color_list=['lime','blue','red','orange','magenta']
for i,n_sample in enumerate([1,2,10]):
    model.fit(np.random.normal(loc=0.8,scale=0.1,size=n_sample))
    ax.plot(x,model.pdf(x),color=color_list[i],label="N={0}".format(n_sample))
plt.xlim(-1,1)
plt.xticks([-1,0,1])
plt.yticks([0,5])
plt.legend(fontsize=12)


'''Maximum likelihood'''
ax = fig.add_subplot(1,3,3)
plt.title('Maximum likelihood')
model.set_param(ave = 0.3, var = 0.05)
ax.plot(x,model.pdf(x),label='population',color='k',zorder=3,linestyle='dashed')

for i,n_sample in enumerate([2,1000]):
    model.fit(np.random.normal(loc=0.3,scale=np.sqrt(0.05),size=n_sample))
    ax.plot(x,model.pdf(x),color=color_list[i+3],label="N={0}".format(n_sample))
plt.xlim(-1,1)
plt.xticks([-1,0,1])
plt.yticks([0])
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


'''Bayesian inference about variance'''
ax = fig.add_subplot(1,3,2)
x = np.linspace(0,3.5,1000)
plt.title('Bayesian inference about ' + r"$1/\sigma^2$")
gamma = Gamma(1., 1.)
model.set_param(ave = 4.,var = gamma)
ax.plot(x,model.var_pdf(x),color='k',label="N=0")

for i,n_sample in enumerate([1,2,10]):
    model.fit(np.random.normal(loc = 4.,scale = 1., size = n_sample))
    ax.plot(x,model.var_pdf(x),color = color_list[i], label = "N={0}".format(n_sample))
plt.xlim(0,3.5)
plt.xticks([0,1,2,3])
plt.yticks([0])
plt.legend(fontsize=12, loc='upper right')
