import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def f(x,y):# Beal's function
    term_1 = np.square(1.5 - x + x * y)
    term_2 = np.square(2.25 - x + x * y ** 2)
    term_3 = np.square(2.625 - x + x * y ** 3)
    return term_1 + term_2 + term_3

xmin, xmax = -4.5, 4.5
ymin, ymax = -4.5, 4.5
X, Y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
Z = f(X,Y)
minimum = np.array([3.,0.5])

fig, ax = plt.subplots(figsize=(10, 6))
ax.contour(X,Y,Z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap='jet')
ax.plot(*minimum, 'r*', markersize=18)
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
plt.show()
