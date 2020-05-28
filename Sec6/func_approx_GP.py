import numpy as np
import matplotlib.pyplot as plt
from kernel import *

def make_noise(n_sample=10, std=0.1, margin=[0., 1.]):
    x = np.linspace(*margin, n_sample)
    t = np.sin(2 * np.pi * x) + np.random.normal(scale=std, size=n_sample)
    return x, t

x_train, y_train = make_noise(n_sample=7, margin=[0, 0.7])
x_test = np.linspace(0, 1, 100)
y_test = np.sin(2 * np.pi * x_test)

model = GP_regression(kernel=GaussianKernel(1,1), beta=100.)
fig = plt.figure(figsize=(12,4))

for n,(max_iter,title) in enumerate(zip([0,1000],['before','after'])):
    fig.add_subplot(1,2,n+1)
    model.fit(x_train, y_train, n_iter=max_iter)
    y,y_std = model.predict(x_test, get_std=True)
    plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", label="noise", s=50, linewidth=1.5, zorder=3)
    plt.plot(x_test, y_test, color="lime", label="sin$(2\pi x)$")
    plt.plot(x_test, y, color="r", label="prediction")
    plt.fill_between(x_test, y - y_std, y + y_std, alpha=0.4, color="pink", label="std")
    plt.title(title + ' fitting', fontsize=20)
    plt.ylim([-1.5,1.5])
    plt.show()
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left', borderaxespad=0, fontsize=15)
plt.tight_layout()
