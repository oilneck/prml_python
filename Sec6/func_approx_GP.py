import numpy as np
import matplotlib.pyplot as plt
from kernel import *

def create_toy_data(func, n=10, std=1., domain=[0., 1.]):
    x = np.linspace(domain[0], domain[1], n)
    t = func(x) + np.random.normal(scale=std, size=n)
    return x, t

def sinusoidal(x):
        return np.sin(2 * np.pi * x)

x_train, y_train = create_toy_data(sinusoidal, n=10, std=0.1)
x_train, y_train = x_train[:7], y_train[:7]
x = np.linspace(0, 1, 100)

model = GP_regression(kernel=GaussianKernel(2,2), beta=100.)
model.fit(x_train, y_train, n_iter=10000)

y,y_std = model.predict(x, get_std=True)
plt.scatter(x_train,y_train,facecolor="none", edgecolor="b",label="noise",s=50,linewidth=1.5)
plt.plot(x, sinusoidal(x), color="lime", label="sin$(2\pi x)$")
plt.plot(x, y, color="r", label="gpr")
plt.fill_between(x, y - y_std, y + y_std, alpha=0.4, color="pink", label="std")
plt.show()
