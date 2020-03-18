import numpy as np
import matplotlib.pyplot as plt
from pd import *


model = Gauss(mu=0.,var=0.1)
x = np.linspace(-1,1,1000)
plt.plot(x,model.pdf(x),color='k',label="N=0")

color_list=['lime','blue','red']
for i,n_sample in enumerate([1,2,10]):
    model.mu_fit(np.random.normal(loc=0.8,scale=0.1,size=n_sample))
    plt.plot(x,model.pdf(x),color=color_list[i],label="N={0}".format(n_sample))


plt.xlim(-1,1)
plt.ylim(0,5)
plt.xticks([-1,0,1])
plt.yticks([0,5])
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()
