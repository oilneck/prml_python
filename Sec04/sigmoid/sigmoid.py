import numpy as np
import matplotlib.pyplot as plt
from base_module.sigmoid_feature import Sigmoid_Feature
from scipy.special import erf


F_size = 15

def probit_inv(a):#probit regression
    return 0.5 + 0.5 * erf(a / np.sqrt(2))

# create sigmoid data
feature = Sigmoid_Feature(mean=[0.],std=1.)
x = np.arange(-10, 10, 0.01)
y = feature.transform(x)[:,1]

'''create probit data'''
cdf_y = probit_inv(x * np.sqrt(np.pi / 8))


# plotting
plt.axvline(x=0, ymin=0, ymax=1, linewidth=1,linestyle='--',color = 'k')
plt.axhline(y=0.5,xmin=-10,xmax=10,linewidth=1,linestyle='--',color = 'k')
plt.plot(x, y,color='r',zorder=1,linewidth=2,label='sigmoid')
plt.plot(x,cdf_y,color='b',linestyle='dashed',zorder=2,linewidth=2,label='probit_inv')
plt.xlim(-7,7)
plt.ylim(-0.1,1.1)
plt.xticks([-5,0,5],fontsize=F_size)
plt.yticks([0,0.5,1],fontsize=F_size)
plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1, fontsize=F_size)
plt.show()
