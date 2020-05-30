import numpy as np
import matplotlib.pyplot as plt
from kernel import *

n_sample = 50

def generate_noise_data(N):
    correlation= [0.9, 0.5]
    cov = [correlation, np.flip(correlation)]
    cls1 = np.random.multivariate_normal([-0.5,1], cov, int(N)).T
    cls2 = np.random.multivariate_normal([1,-0.5], cov, int(N)).T
    t = np.hstack((np.ones(cls1.shape[1]),np.zeros(cls2.shape[1])))
    return np.c_[cls1, cls2].T, t



X_train, t_train = generate_noise_data(n_sample)
X,Y = np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
X_test = np.array([X.ravel(), Y.ravel()]).reshape(2,-1).T


''' Learning '''
model = RVM_classifier(kernel=GaussianKernel(1, 0.5))
model.fit(X_train, t_train)
Z = model.predict(X_test)

# plotting training data
plt.scatter(*X_train.T, c=np.where(t_train, 'r', 'b'), marker='x', s=50, linewidth=1.5)
plt.scatter(*model.relevance_vector['x'].T, s=130, facecolor="none", edgecolor='limegreen', linewidth=1.5)

# plotting test data
plt.contour(X, Y, Z.reshape(X.shape), alpha=1, levels=np.linspace(0, 1, 3), cmap='Greys', linestyles='--')
plt.contourf(X, Y, Z.reshape(X.shape), alpha=0.2, levels=np.linspace(0, 1, 5), cmap='jet')
plt.colorbar()
plt.show()
