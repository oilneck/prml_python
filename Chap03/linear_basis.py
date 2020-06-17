import numpy as np
from base_module import *
import matplotlib.pyplot as plt

deg = 11

x = np.linspace(-1,1,100)
polynomial_basis = Poly_Feature(deg).transform(x)
gaussian_basis = Gaussian_Feature(np.linspace(-1,1,deg),0.1).transform(x)
sigmoid_basis = Sigmoid_Feature(np.linspace(-1,1,deg),0.1).transform(x)

fig = plt.figure(figsize=(15, 5))
for i,basis in enumerate([polynomial_basis,gaussian_basis,sigmoid_basis],1):
    fig.add_subplot(1,3,i)
    for col in range(deg+1):
        plt.plot(x,basis[:,col])
        plt.xlim([-1,1])
        plt.xticks([-1,0,1])
plt.tight_layout()
plt.show()
