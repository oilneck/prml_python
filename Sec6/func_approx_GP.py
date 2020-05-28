import numpy as np
import matplotlib.pyplot as plt
from kernel import *

def create_toy_data(func, n=7, std=0.1, domain=[0., 0.7]):
    x = np.linspace(domain[0], domain[1], n)
    t = func(x) + np.random.normal(scale=std, size=n)
    return x, t

def sinusoidal(x):
        return np.sin(2 * np.pi * x)

x_train, y_train = create_toy_data(sinusoidal, std=0.1)
x = np.linspace(0, 1, 100)

model = GP_regression(kernel=GaussianKernel(1,1), beta=100.)
fig = plt.figure(figsize=(12,4))

for n,max_iter in enumerate([0,500],1):
    fig.add_subplot(1,2,n)
    model.fit(x_train,y_train,n_iter=max_iter)
    y,y_std = model.predict(x, get_std=True)
    plt.scatter(x_train,y_train,facecolor="none", edgecolor="b",label="noise",s=50,linewidth=1.5)
    plt.plot(x, sinusoidal(x), color="lime", label="sin$(2\pi x)$")
    plt.plot(x, y, color="r", label="prediction")
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.4, color="pink", label="std")
    plt.show()
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left', borderaxespad=0, fontsize=15)
plt.tight_layout()
