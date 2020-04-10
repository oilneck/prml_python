import numpy as np
from deepL_module.base.functions import *
import matplotlib.pyplot as plt

act_func = [
            'identity_function',
            'step_function',
            'tanh',
            'sigmoid',
            'relu',
            'elu',
            'softsign',
            'softplus',
            'swish'
            ]

x = np.linspace(-3, 3, 500)


style = ['-', '--', ':', '-.', '-', '-', '-', '--', '-']
markers = ["", "", "", "", "+", "x", ".", "*", ""]

fig = plt.figure(figsize=(13,8),dpi=50)
ax = fig.add_subplot(121)
cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
cmap.reverse()

for n, name in enumerate(act_func):
    y = eval(name + '(x)')
    ax.plot(x, y,
            label=name.split('_')[0],
            linestyle = style[n],
            marker = markers[n],
            color = cmap[n],
            markersize = 7,
            alpha = 1 - 0.05 * n,
            zorder = n + 1,
            markevery = 30)
plt.tick_params(labelsize=12)
plt.title('Common activation functions', fontsize=20)
plt.legend(fontsize=15)
plt.grid()

ax = fig.add_subplot(122)
cmap.remove(cmap[0])
markers = markers[::-1]

y = swish(x, beta=1.)
def order_n(i): return {1:"1st", 2:"2nd", 3:"3rd"}.get(i) or "%dth"%i
for n in range(3):
    ax.plot(x, y,
            color = cmap[n],
            marker = markers[n],
            markersize = 10,
            markevery = 30,
            label = order_n(n) + ' derivatives')
    y = np.gradient(y,x)
plt.vlines(x=0, ymin=-0.75, ymax=7, linewidth=1,linestyle='-',color = 'r')
plt.hlines(y=0, xmin=-4, xmax=4, linewidth=1,linestyle='-',color = 'r')
plt.ylim(bottom=-0.4,top=1.2)
plt.xlim(left=-3.2, right=3.2)
plt.tick_params(labelsize=12)
plt.title('swish', fontsize=20)
plt.legend(fontsize=15)
plt.grid()
plt.show()
