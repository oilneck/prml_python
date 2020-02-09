import numpy as np
import matplotlib.pyplot as plt
from fitting.bayesian_regression import Bayesian_Regression
from base_module.poly_feature import Poly_Feature

Font_size = 15
M = 15
feature = Poly_Feature(M)


def kernel_function(train_x,test_x):
    model = Bayesian_Regression(degree=M,beta=5)
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
plt.plot(x,y,color='b')
plt.xlabel(r"$x$",fontsize=Font_size)
plt.ylabel(r"$k(0,x)$",fontsize=Font_size)
plt.xticks([-1,0,1],fontsize=Font_size)
plt.yticks([0,0.02,0.04],fontsize=Font_size)
plt.xlim(-1,1)
plt.ylim(-0.01,0.04)
plt.hlines([0], -1, 1, "black", linestyles='dotted',linewidth=1)
plt.text(0,0,r"$\times$", size = 20, color = "red",horizontalalignment="center", verticalalignment='center')
plt.tight_layout()
plt.show()
