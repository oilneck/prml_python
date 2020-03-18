import numpy as np
import matplotlib.pyplot as plt
from fitting.bayesian_regression import Bayesian_Regression
from base_module import *

Font_size = 15
M = 20
feature = Gaussian_Feature(np.linspace(-1.,1.,M),0.1)
model = Bayesian_Regression(degree=M,beta=10)
model.feature = feature

def kernel_function(train_x,test_x):
    model.fit(test_x,np.zeros(len(test_x)))
    phi1 = feature.transform(train_x)
    phi2 = feature.transform(test_x)
    return model.beta * np.sum(phi1 * (model.w_cov @ phi2.T).T,axis=1)


# test data & dummy training data
x = np.arange(-1.01, 1.01, 0.01)
train_x = np.zeros(len(x))
y = kernel_function(train_x,x)


#plotting kernel function
plt.close('all')
fig = plt.figure(figsize=(7,3))
ax = fig.add_subplot(1,2,1)
ax.plot(x,y,color='b')
plt.xlabel(r"$x$",fontsize=Font_size)
plt.ylabel(r"$k(0,x)$",fontsize=Font_size)
plt.xticks([-1,0,1],fontsize=Font_size)
plt.yticks([0,0.02,0.04],fontsize=Font_size)
plt.xlim(-1,1)
plt.ylim(-0.01,0.04)
plt.hlines([0], -1, 1, "black", linestyles='dotted',linewidth=1)
plt.text(0,0,r"$\times$", size = 20, color = "red",horizontalalignment="center", verticalalignment='center')

# drawing kernel 2D function
X,Y = np.meshgrid(np.linspace(-1.1,1.1,100),np.linspace(-1.1,1.1,100))
Z = kernel_function(X.ravel(),Y.ravel())
ax = fig.add_subplot(1,2,2)
ax.contourf(X,Y,Z.reshape(X.shape),levels=np.linspace(min(Z),max(Z)/2.7,200),cmap='jet')
ax.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
ax.set_aspect('equal')
plt.xlabel(r"$k(x,x')$",fontsize=Font_size)
plt.xlim(-1.,1.)
plt.ylim(-1.,1.)
plt.subplots_adjust(wspace=1.5)
plt.tight_layout()
plt.show()
