import numpy as np
import matplotlib.pyplot as plt
from kernel import *

def generate_noise_data(N):
    correlation= [1., 0.5]
    cov = [correlation, np.flip(correlation)]
    N = int(N // 2)
    cls1 = np.random.multivariate_normal([-0.5,1], cov, int(N)).T
    cls2 = np.random.multivariate_normal([1,-0.5], cov, int(N)).T
    t = np.hstack((np.ones(cls1.shape[1]), -np.ones(cls2.shape[1])))
    return np.c_[cls1, cls2].T, t



# training data & test data
X_train, t_train = generate_noise_data(50)
X,Y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
X_test = np.array([X, Y]).reshape(2, -1).T

''' learning '''
model = C_SVM(kernel=GaussianKernel(*[1,0.5,0.5]), C=1.)
model.fit(X_train ,t_train, n_iter=100)
Z = model.predict(X_test)

# plot training data
plt.scatter(*X_train.T, c=np.where(t_train > 0, 'r', 'b'), marker='x', s=50, linewidth=1.5, zorder=5)

# draw prediction data
plt.contour(X, Y, Z.reshape(X.shape), np.array([-1,0,1]), colors="k", linestyles=("dashed","solid","dashed"))
plt.scatter(*model.support_vector['x'].T, s=130, facecolor="none", edgecolor='limegreen', linewidth=1.5)
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
