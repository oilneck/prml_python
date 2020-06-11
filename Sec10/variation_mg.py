import numpy as np
import matplotlib.pyplot as plt
from pd import *


def make_blobs():
    cls1 = np.random.normal(size=(100, 2), loc=(-5,-5))
    cls2 = np.random.normal(size=(100, 2), loc=(5, -5))
    cls3 = np.random.normal(size=(100, 2), loc=(0, 5))
    return np.vstack((cls1, cls2, cls3))


# training & test data
X_train = make_blobs()
x_test, y_test = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
X_test = np.array([x_test, y_test]).reshape(2, -1).T

''' variational reasoning '''
vbgm = BayesianGaussianMixture(n_components=10, alpha_0=0.01)
vbgm.fit(X_train)
Z = vbgm.predict(X_test)


plt.contour(x_test, y_test, Z.reshape(x_test.shape))
plt.scatter(X_train[:, 0], X_train[:, 1], cmap='jet', c=vbgm.classify(X_train))
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
