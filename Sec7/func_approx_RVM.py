import numpy as np
import matplotlib.pyplot as plt
from kernel import *

def func(x):
    return np.sin(2 * np.pi * x)

def generate_noise_data(n_sample):
    x = np.linspace(0, 1, n_sample)
    t = func(x) + np.random.normal(scale=0.2, size=n_sample)
    return x,t


x_train, y_train = generate_noise_data(10)
x = np.linspace(0, 1, 100)


'''---- Learning ----'''
model = RVM_regression(kernel=GaussianKernel(1., 20.))
model.fit(x_train, y_train)
y, y_std = model.predict(x, get_std=True)


# plot the test data
plt.plot(x, y, color='red', label="predict mean")
plt.fill_between(x, y + y_std, y - y_std, facecolor='pink', alpha=0.4, label="std.")

# plot the training data
plt.scatter(x_train, y_train, facecolor="none", edgecolor="limegreen", label="noise", s=50, linewidth=1.5)
plt.scatter(*model.relevance_vector.values(), s=130, facecolor="none", edgecolor="b", label="relevance vector")

# config for drawing
plt.legend(fontsize=12)
plt.xlim(-0.05, 1.05)
plt.ylim(-1.7, 1.7)
plt.xticks([0,1])
plt.yticks([-1,0,1])
plt.show()
