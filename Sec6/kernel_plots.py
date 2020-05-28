import numpy as np
import matplotlib.pyplot as plt
from kernel import *

fig = plt.figure(figsize=(14,7))
div_num = 70
x = np.linspace(-1,1,div_num)
params = [[1.,4.,0.,0.],[9.,4.,0.,0.],[1.,64.,0.,0.],
          [1.,0.25,0.,0.],[1.,4.,10.,0],[1.,4.,0.,5.]]

for n, param in enumerate(params):
    fig.add_subplot(2,3,n+1)
    kernel = GaussianKernel(*param)
    model = GP_regression(kernel, beta=10.)
    model.fit(x,x,n_iter=0)
    for _ in range(1,6):
        gram = np.copy(model.Gram)
        y = np.random.multivariate_normal(np.zeros(div_num), gram, 1).ravel()
        plt.plot(x,y)
        plt.title('{}'.format(param),fontsize=15)
        plt.xlim([-1,1])
        plt.xticks(np.arange(-1,1.1,0.5),fontsize=10)
        plt.tight_layout()
plt.show()
