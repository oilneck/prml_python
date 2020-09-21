import numpy as np
import matplotlib.pyplot as plt
from base_module import Poly_Feature
from fitting import *

max_deg = 9

def generate_noise_data(func, sample_size, std, domain=[0, 1]):
    x = np.linspace(domain[0], domain[1], sample_size)
    np.random.shuffle(x)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


def func(x):
    return  x * (x - 2) * (x - 3)


# training data & test data
train_x, train_t = generate_noise_data(func, 7, 10, [-6, 5])
x = np.linspace(-6, 6, 100)


'''Variational lower bound'''
L_list = []
models = []
for m in range(1, max_deg + 1):
    model = VariationalRegressor(b0=10, d0=100)
    X_train = Poly_Feature(m).transform(train_x)
    model.fit(X_train, train_t)
    l_bound = model.lower_bound()
    L_list.append(l_bound)
    models.append(model)


L_list = np.array(L_list)
idx = np.argmax(L_list)

# plot lower bound
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(121)
ax.plot(np.arange(1, max_deg + 1), L_list, color='b', marker='o', markeredgecolor='b', markerfacecolor='w', markersize=7)
plt.xlabel("degree", fontsize=20)
plt.title('Variational lower bound',fontsize=17)


# plot training data
ax = fig.add_subplot(122)
ax.scatter(train_x, train_t, facecolor="none", edgecolor='b', label='noise')
ax.plot(x, func(x), color='limegreen', label='model')

# plot optimal prediction
X_test = Poly_Feature(idx+1).transform(x)
y_mean, y_std = models[idx].predict(X_test, get_std=True)

plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.4, color="pink", label='std')
ax.text(0.9,0.2,"optimal degree: {}".format(idx+1),ha='right',va='top', transform=ax.transAxes, fontsize=17)
plt.plot(x, y_mean, c="r", label="prediction")
ax.legend(bbox_to_anchor=(1.05,0.5),loc='upper left',borderaxespad=0,fontsize=15)
plt.tight_layout()
plt.show()
