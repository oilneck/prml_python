import numpy as np
from pd import *
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
fig = plt.figure(figsize=(7,4))
for n,[a,b] in enumerate([[0.1,0.1],[1.,1.],[2.,3.],[8.,4.]],1):
    prob = Beta(a,b)
    ax = fig.add_subplot(2,2,n)
    ax.plot(x,prob.pdf(x),color='r')
    plt.xlabel(r"$\mu$")
    plt.xticks([0,0.5,1])
    plt.yticks(np.arange(0,4,1))
    plt.xlim(0,1)
    plt.ylim(0,3)
    plt.annotate("a={:.1g}".format(a), (0.1, 2.5))
    plt.annotate("b={:.1g}".format(b), (0.1, 1.9))
plt.tight_layout()
plt.show()
