import numpy as np
import matplotlib.pyplot as plt
from kernel import *

def create_noise_data(n_sample=10, std=0.1, margin=[0, 1]):
    x = np.linspace(*margin, n_sample)
    t = np.sin(2 * np.pi * x) + np.random.normal(scale=std, size=n_sample)
    return x, t

x_train, y_train = create_noise_data(n_sample=10)
x = np.linspace(0, 1, 100)
y_test = np.sin(2 * np.pi * x)

model = GP_regression(kernel=PolynomialKernel(3,1), beta=1e10)
model.fit(x_train, y_train)
y = model.predict(x)

fig = plt.figure(figsize=(7,4))
fig.add_subplot(111)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", label="noise", s=50, linewidth=1.5, zorder=3)
plt.plot(x, y_test, color="lime", label="sin$(2\pi x)$")
plt.plot(x, y, color="r", label="prediction")
plt.title('Function approxi. with Polynomial Kernel', fontsize=15)
plt.tight_layout()
plt.show()
