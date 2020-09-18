import matplotlib.pyplot as plt
import numpy as np
from deepL_module.datasets import *
from deepL_module.base import *
from pd import *


def ellipse_draw(ax, mu, var):
    for center, cov in zip(mu, var):
        X, Y = ellipse2D_orbit(center, cov).T
        ax.plot(X, Y, color='r', linewidth=3)


# training data & test data
X_train = load_faithful(normalize=True)
x_test, y_test = np.meshgrid(np.linspace(0, 6, 100), np.linspace(0, 100, 100))
X_test = np.array([x_test, y_test]).reshape(2, -1).T


''' Variational Gaussian Mixture '''
model = BayesianGaussianMixture(n_components=4, alpha_0=1e-2)
model.fit(X_train, n_iter=100)
labels = model.classify(X_train)
Z = model.predict(X_test)



fig = plt.figure(figsize=(17,4))
keys = list(model.eff_params.keys())
n_step = len(keys)
keys = [keys[0], keys[n_step // 4], keys[2 * n_step // 3], keys[-1]]
for n, key in enumerate(keys):
    ax_ = fig.add_subplot(1, len(keys), n+1)
    ax_.scatter(*X_train.T, c='limegreen')
    ax_.scatter(*model.eff_params[key]['means'].T, s=130, marker='X', lw=1, c='r', edgecolor="white", zorder=3)
    ellipse_draw(ax_, model.eff_params[key]['means'], model.eff_params[key]['covs'])
    ax_.set_title(str(key), fontsize=20)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.tick_params(bottom=False, left=False)
plt.show()
