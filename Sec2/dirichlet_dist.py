import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pd.dirichlet import Dirichlet


'''triangle mesh grid (0,0)-(1,0)-(0,1)'''
X_test = np.array([[0.01*a*0.01*(100-b) for a in range(1, 100)] for b in range(1, 100)])
Y_test = np.array([[0.01*b] * 99 for b in range(1, 100)])
mesh_data = np.array([X_test.ravel(),Y_test.ravel(),1-X_test.ravel()-Y_test.ravel()]).T

X = np.array([x + (0.5 - np.average(x)) for x in X_test])
Y = Y_test * np.sqrt(3) / 2

fig = plt.figure(figsize=(10,3))
for i,alpha in enumerate([0.1,1,10],1):
    ax = fig.add_subplot(1, 3, i, projection='3d')
    prob = Dirichlet(np.repeat(alpha,3))
    Z = prob.pdf(mesh_data)
    ax.plot_surface(X,Y, Z.reshape(X.shape),cmap='rainbow')
    ax.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
plt.show()
