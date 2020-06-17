import matplotlib.pyplot as plt
import numpy as np
from pd import *
from scipy.stats import norm, gamma


def pred(mu, tau, params):
    Norm = norm.pdf(mu, params[0], np.sqrt(1 / params[1]))
    Gamma = gamma.pdf(tau, a=params[2], scale= 1 / params[3])
    return Norm * Gamma

true_mu = 0
true_tau = 0.5


# preparing training data
X_train = np.random.normal(true_mu, np.sqrt(1. / true_tau), 75)
mu_test, lam_test = np.meshgrid(np.linspace(-1,1,100), np.linspace(0.01,2,100))


'''Variational estimation'''
model = VariationalGaussian(*[.5, 7, 4, 25])
model.fit(X_train)
pdf = model.pdf(mu_test.ravel(), lam_test.ravel())


fig = plt.figure(figsize = (17, 4))
n_step = len(model.history) // 2
keys = [
        model.history['tauStep1'],
        model.history['muStep2'],
        model.history['tauStep2'],
        model.history['tauStep' + str(n_step)]
        ]
c = ['b', 'b', 'b', 'r']
text = ['init', 'estimate abut $\mu$', r'estimate about $\tau$', 'optimal solution']
for n, key in enumerate(keys):
    ax = fig.add_subplot(1, 4, n+1)
    ax.contour(mu_test, lam_test, pdf.reshape(mu_test.shape), levels=np.linspace(min(pdf), max(pdf), 7), colors='limegreen')
    plt.xticks([0], fontsize=15)
    plt.yticks([0.5], fontsize=15)
    plt.ylim(0,1)
    plt.hlines(y=true_tau, xmin=-1, xmax=1, linewidth=1, linestyle='--', color = 'k')
    plt.vlines(x=true_mu, ymin=-1, ymax=1, linewidth=1, linestyle='--', color = 'k')
    Z = pred(mu_test.ravel(), lam_test.ravel(), key)
    ax.contour(mu_test, lam_test, Z.reshape(mu_test.shape), levels=np.linspace(min(Z), max(Z), 7), colors=c[n])
    plt.title(text[n], fontsize=15)
plt.tight_layout()
plt.show()
