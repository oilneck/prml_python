import numpy as np
import matplotlib.pyplot as plt
from pd import *

mu=np.linspace(0,1,1000)



fig, ax = plt.subplots(1,3,figsize=(10,2))
model = [Beta(2,2),Binomial(trials=1,prob=mu),Beta(2,2)]
color_list = ['r','b','r']
title_list = ["prior","likelihood function","posterior"]
for i,x in enumerate([mu,1,mu]):
    ax[i].set_title(title_list[i])
    if title_list[i] == "posterior":
        model[i].fit([1])
    ax[i].plot(mu,model[i].pdf(x).ravel(),color=color_list[i])
    ax[i].set_xlim([0,1])
    ax[i].set_ylim([0,2])
    ax[i].set_xticks([0,1])
    ax[i].set_yticks([0,1,2])

plt.gcf().text(0.35,0.45,r"$\times$",fontsize=20,color='orange')
plt.gcf().text(0.64,0.45,"=",fontsize=20,color='orange')
plt.subplots_adjust(wspace=0.5)
plt.show()
