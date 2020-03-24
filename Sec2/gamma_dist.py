import numpy as np
from pd import *
import matplotlib.pyplot as plt

x = np.linspace(0,2,100)
fig = plt.figure(figsize=(11.,3.))
# plot gamma distribution
for n,[a,b] in enumerate([[0.1,0.1],[1.,1.],[4.,6.]],1):
    prob = Gamma(a,b)
    ax = fig.add_subplot(1,4,n)
    ax.plot(x,prob.pdf(x),color='r')
    plt.title("Gam"+r"$(\lambda|a,b)$",fontsize=9)
    plt.xlabel(r"$\lambda$")
    plt.xticks([0,0.5,1])
    plt.yticks(np.arange(0,4,1))
    plt.xlim(0,2)
    plt.ylim(0,2)
    plt.xticks(np.arange(0,3,1))
    plt.annotate("a={:.1g}".format(a), (1.2, 1.6))
    plt.annotate("b={:.1g}".format(b), (1.2, 1.2))


# plot nomal-gamma distribution
ax = fig.add_subplot(144)
mu,lamda = np.meshgrid(np.linspace(-2,2,100),np.linspace(0,2,100))
prob = Gamma(5,6)
Z = prob.norm_gamma(mu=mu.ravel(),x=lamda.ravel())
ax.contour(mu,lamda,Z.reshape(mu.shape),levels=np.linspace(min(Z),max(Z),8),cmap='jet')
ax.tick_params(labelleft=False,labelright=True)
plt.xticks([-2,0,2])
plt.yticks([0,2])
plt.xlim(-2,2)
plt.ylim(0,2)
plt.title("normal-gamma dist.",fontsize=9)
plt.xlabel(r"$\mu$")
plt.ylabel(r"$\lambda$")
plt.subplots_adjust()
plt.tight_layout()
plt.show()
