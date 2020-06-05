import numpy as np
import matplotlib.pyplot as plt
from pd import *


def make_blobs():
    x1 = np.random.normal(size=(100, 2)) + np.array([-5, -5])
    x2 = np.random.normal(size=(100, 2)) + np.array([0, 5])
    x3 = np.random.normal(size=(100, 2)) + np.array([5, -5])
    return np.vstack((x1, x2, x3))


# training data & test data
X_train = make_blobs()
X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
X_test = np.array([X, Y]).reshape(2, -1).T


''' ML estimation with EM-algorithm '''
prob_dist = MultivariateGaussian(n_components = 3)
prob_dist.fit(X_train)
labels = prob_dist.classify(X_train)
Z = prob_dist.predict(X_test)


# plot training data & prediction data
fig = plt.figure(figsize=(12,4))
keys = list(prob_dist.centers.keys())
n_step = len(keys)
keys = [keys[0], keys[n_step // 3], keys[2 * n_step // 3], keys[-1]]
c=['dodgerblue', 'r', 'limegreen']
for n, key in enumerate(keys):
    fig.add_subplot(1, len(keys), n+1)
    plt.scatter(*X_train.T, c=[c[int(label)] for label in labels], alpha=.5)
    plt.scatter(*prob_dist.centers[key].T, s=170, marker='X', lw=2, c=c, edgecolor="white", zorder=3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.tick_params(bottom=False,left=False)
plt.contour(X, Y, Z.reshape(X.shape), levels=np.linspace(min(Z), max(Z), 5), linewidths=4, cmap='Greys_r')
plt.show()
