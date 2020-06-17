import numpy as np
import matplotlib.pyplot as plt
from base_module import Poly_Feature
from fitting import *

max_deg = 9
margin = 5

def create_toy_data(func, sample_size, std, domain=[0, 1]):
    x = np.linspace(domain[0], domain[1], sample_size)
    np.random.shuffle(x)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t

def func(x):
    return  x ** 2 * (x - 3)

'''
x ** 2 * (x - 3)

noiseNUM = 10
noise_std = 20 [-5, 5]

beta = 5e-3
a0 = b0 = 1
'''


train_x, train_t = create_toy_data(func, 10, 15, [-6, 5])



L_list = []
models = []
for m in range(1, max_deg + 1):
    model = VariationalRegressor(beta=5e-3, a0=1, b0=1)
    X_train = Poly_Feature(m).transform(train_x)
    model.fit(X_train, train_t)
    l_bound = model.lower_bound(X_train, train_t)
    L_list.append(l_bound)
    models.append(model)

L_list = np.array(L_list)
idx = np.argmax(L_list)

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(121)


ax.plot(np.arange(1, max_deg + 1), L_list, color='b')
ax.scatter(np.arange(1, max_deg + 1), L_list, edgecolor='b', facecolor='none', s=50)
plt.xlabel("degree", fontsize=15)


ax = fig.add_subplot(122)
ax.scatter(train_x, train_t, facecolor="none", edgecolor='b')
x = np.linspace(-6, 6, 100)
ax.plot(x, func(x), color='limegreen')
X_test = Poly_Feature(idx+1).transform(x)
y_mean, y_std = models[idx].predict(X_test, get_std=True)
plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.4, color="pink")
plt.plot(x, y_mean, c="r", label="prediction")
plt.tight_layout()
plt.show()
