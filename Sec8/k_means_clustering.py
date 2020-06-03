import numpy as np
import matplotlib.pyplot as plt
from clustering import *

# training data
x1 = np.random.normal(size=(100, 2))
x1 += np.array([-5, -5])
x2 = np.random.normal(size=(100, 2))
x2 += np.array([5, -5])
x3 = np.random.normal(size=(100, 2))
x3 += np.array([0, 5])
X_train = np.vstack((x1, x2, x3))

X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
X_test = np.array([X, Y]).reshape(2, -1).T

model = Kmeans(n_clusters=3)
model.fit(X_train)
Z = model.predict(X_test)

plt.scatter(X_train[:, 0], X_train[:, 1], c=model.predict(X_train))
plt.scatter(*model.means.T, s=200, marker='X', lw=2, c=['purple', 'cyan', 'yellow'], edgecolor="white")
plt.contourf(X, Y, Z.reshape(X.shape), alpha=0.1)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
