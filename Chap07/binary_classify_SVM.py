import numpy as np
import matplotlib.pyplot as plt
from kernel import *

def make_circles(n_sample:int, noise, factor=.8, margin=[0, np.pi / 2]):
    N_in = int(n_sample // 3)
    N_out = n_sample - N_in
    _out = np.linspace(margin[0] - 0.05 * np.pi, margin[1] + 0.05 * np.pi, N_out)
    _in = np.linspace(margin[0], margin[1], N_in)
    bias = 0.2
    out_x, out_y = np.cos(_out)+bias, np.sin(_out)+bias
    in_x, in_y = np.cos(_in)-bias, np.sin(_in)-bias
    x = np.vstack([np.append(factor * out_x, 0.9 * in_x),
                   np.append(factor * out_y, 0.9 * in_y)]).T
    x += np.random.normal(scale=noise, size=x.shape)
    t = np.hstack([-np.ones(N_out), np.ones(N_in)])
    return x, t




# training data & test data
X_train, t_train = make_circles(80, factor=1.6, noise=0.2)
X,Y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
X_test = np.array([X, Y]).reshape(2, -1).T

''' learning '''
model = C_SVM(kernel=GaussianKernel(*[1,1,1]))
model.fit(X_train ,t_train, n_iter=1000)
Z = model.predict(X_test)

# plot training data
plt.scatter(*X_train.T, c=np.where(t_train > 0, 'r', 'b'), marker='x', s=50, linewidth=1.5, zorder=5)

# draw prediction data
plt.contour(X, Y, Z.reshape(X.shape), np.array([-1,0,1]), colors="k", linestyles=("dashed","solid","dashed"))
plt.scatter(*model.support_vector['x'].T, s=130, facecolor="none", edgecolor='limegreen', linewidth=1.5)
plt.xlim([-0.2, 2.5])
plt.ylim([-0.3, 2.3])
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
