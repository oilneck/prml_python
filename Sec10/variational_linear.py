import numpy as np
import matplotlib.pyplot as plt
from base_module import Poly_Feature
from fitting import *

def create_toy_data(func, sample_size, std, domain=[0, 1]):
    x = np.linspace(domain[0], domain[1], sample_size)
    np.random.shuffle(x)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t

def cubic(x):
    return x * (x - 5) * (x + 5)


feature = Poly_Feature(degree=3)
train_x, train_t = create_toy_data(cubic, 10, 10., [-5, 5])
X_train = feature.transform(train_x)

x = np.linspace(-5, 5, 100)
X_test = feature.transform(x)


model = VariationalRegressor(beta=1e-2, a0=1, b0=1)
model.fit(X_train, train_t)
y_mean, y_std = model.predict(X_test, get_std=True)


plt.scatter(train_x, train_t, s=100, facecolor="none", edgecolor="b")
plt.plot(x, cubic(x), c="g", label="$\sin(2\pi x)$")
plt.plot(x, y_mean, c="r", label="prediction")
plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color="pink")
plt.legend()
plt.show()
