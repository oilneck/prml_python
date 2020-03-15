import numpy as np
from pd import *
import matplotlib.pyplot as plt

x = np.linspace(0,2,100)
fig = plt.figure(figsize=(7,2.5))
for n,[a,b] in enumerate([[0.1,0.1],[1.,1.],[4.,6.]],1):
    prob = Gamma(a,b)
    ax = fig.add_subplot(1,3,n)
    ax.plot(x,prob.pdf(x),color='r')
    plt.xlabel(r"$\lambda$")
    plt.xticks([0,0.5,1])
    plt.yticks(np.arange(0,4,1))
    plt.xlim(0,2)
    plt.ylim(0,2)
    plt.xticks(np.arange(0,3,1))
    plt.annotate("a={:.1g}".format(a), (1.2, 1.6))
    plt.annotate("b={:.1g}".format(b), (1.2, 1.4))
plt.tight_layout()
plt.show()
