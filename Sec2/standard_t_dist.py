import numpy as np
import matplotlib.pyplot as plt
from pd import *

prob = Students_t(mu=0.,tau=1.)

x = np.linspace(-5,5,1000)

fig = plt.figure(figsize=(7,3))
ax = fig.add_subplot(111)
color_list = ['red','blue','lime']
for i,nu in enumerate([0.1,1.,300]):
    prob.set_df(nu)
    ax.plot(x,prob.pdf(x),color=color_list[i],label=r'$\nu={0}$'.format(nu))
plt.xlim(-5,5)
plt.xticks([-5,0,5])
plt.yticks(np.arange(0,0.6,0.1))
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()
