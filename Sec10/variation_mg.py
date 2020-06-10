import numpy as np
import matplotlib.pyplot as plt
from pd import *


def create_toy_data():
    x1 = np.random.normal(size=(100, 2), loc=(-5,-5))
    x2 = np.random.normal(size=(100, 2), loc=(5, -5))
    x3 = np.random.normal(size=(100, 2), loc=(0, 5))
    return np.vstack((x1, x2, x3))




X_train = create_toy_data()
model = Variational_MG(n_components=10, alpha_0=0.01)
model.fit(X_train)


x_test, y_test = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
X_test = np.array([x_test, y_test]).reshape(2, -1).transpose()

Z = model.predict(X_test)
plt.contour(x_test, y_test, Z.reshape(x_test.shape))
labels = model.classify(X_train)
plt.scatter(X_train[:, 0], X_train[:, 1], cmap='jet', c=labels)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
