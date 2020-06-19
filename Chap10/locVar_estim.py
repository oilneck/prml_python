import numpy as np
import matplotlib.pyplot as plt
from deepL_module.base import *

def g(eta):
    return -eta * np.log(eta) - (1 - eta) * np.log(1 - eta)

def upper(x, eta):
    return np.exp(eta * x - g(eta))

def lamb(x):
    return 0.5 * (sigmoid(x) - 0.5) / x

def lower(x, xi):
    return sigmoid(xi) * np.exp(0.5 * (x - xi) - lamb(xi) * (x ** 2 - xi ** 2))


x = np.linspace(-6, 6, 100)


fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(121)
ax.plot(x, sigmoid(x), c='r')
ax.plot(x, upper(x, eta=0.2), c='b', ls='dashed')
ax.plot(x, upper(x, eta=0.7), c='b', ls='dashed')
plt.text(0, .2, r"$\eta$=0.7" ,fontsize=15, ha='center')
plt.text(2, .7, r"$\eta$=0.2" ,fontsize=15, ha='center')
plt.xticks([-6, 0, 6], fontsize=12)
plt.xlim(-6, 6)
plt.ylim(0, 1)


ax = fig.add_subplot(122)
ax.plot(x, sigmoid(x), c='r')
ax.plot(x, lower(x, xi=2.5), c='b', ls='dashed')
ax.vlines(x=-2.5, ymin=0, ymax=sigmoid(-2.5), lw=1, ls='--', color='limegreen')
ax.vlines(x= 2.5, ymin=0, ymax=sigmoid( 2.5), lw=1, ls='--', color='limegreen')
plt.xticks([-6, -2.5, 0, 2.5, 6],("-6", r"$-\xi$" ,"0", r"$\xi$", "6"), fontsize=12)
plt.text(-2.5, .7, r"$\xi$=2.5", fontsize=15, ha='center')
plt.xlim(-6, 6)
plt.ylim(0, 1)

plt.show()
