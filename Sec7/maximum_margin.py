import numpy as np
import matplotlib.pyplot as plt
from kernel import *

# training data
X_train = np.array([[-3, 3], [0.5, 3.5], [-1, 2.3],
                    [1, 2.5], [-2, 1], [1, -1.5],
                    [3, -2.1], [0, -2.8], [2, -3.5]])

t_train = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1])

# test data
X,Y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
X_test = np.array([X, Y]).reshape(2, -1).T

''' learning '''
model = C_SVM(kernel=PolynomialKernel(degree=1))
model.fit(X_train, t_train)
Z = model.predict(X_test)

# plotting train data
plt.scatter(*X_train.T, c=np.where(t_train > 0, 'r', 'b'), marker='.', s=100, linewidth=1.5, zorder=5)

# plot prediction data
plt.contour(X, Y, Z.reshape(X.shape), np.array([-1, 0, 1]), colors="k", linestyles=("dashed", "solid", "dashed"))
plt.scatter(*model.support_vector['x'].T, s=150, facecolor="none", edgecolor='limegreen', linewidth=2)
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
