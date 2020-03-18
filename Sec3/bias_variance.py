import numpy as np
import matplotlib.pyplot as plt
from fitting.bayesian_regression import Bayesian_Regression
from base_module import *

M = 24

def create_noise_data(sample_size, std, domain=[0, 1]):
    x = np.linspace(domain[0], domain[1], sample_size)
    np.random.shuffle(x)
    t = np.sin(2 * np.pi * x) + np.random.normal(scale=std, size=x.shape)
    return x, t



feature = Gaussian_Feature(np.linspace(0, 1, M), 0.1)

x_test = np.linspace(0, 1, 100)
y_test = np.sin(2 * np.pi * x_test)

for a in [1e2, 1., 1e-9]:
    y_list = []
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    for i in range(100):
        x_train, y_train = create_noise_data(25, 0.25)
        X_train = feature.transform(x_train)
        X_test = feature.transform(x_test)
        model = Bayesian_Regression(degree=M,alpha=a, beta=1.)
        model.feature = feature
        model.fit(X_train, y_train)
        y = model.predict(X_test)
        y_list.append(y)
        if i < 20:
            plt.plot(x_test, y, c="orange")
    plt.ylim(-1.5, 1.5)

plt.subplot(1, 2, 2)
plt.plot(x_test, y_test)
plt.plot(x_test, np.asarray(y_list).mean(axis=0))
plt.ylim(-1.5, 1.5)
plt.show()
