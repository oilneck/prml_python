import numpy as np
from pylab import *
from kernel import *

LR = 0.05
C = np.inf
CountMax = 1000

def generate_noise_data(N):
    correlation= [0.9, 0.5]
    cov = [correlation, np.flip(correlation)]
    N = int(N // 2)
    cls1 = np.random.multivariate_normal([-0.5,1], cov, int(N)).T
    cls2 = np.random.multivariate_normal([1,-0.5], cov, int(N)).T
    t = np.hstack((np.ones(cls1.shape[1]), -np.ones(cls2.shape[1])))
    return np.c_[cls1, cls2].T, t




kernel = GaussianKernel(*[1.,0.5,0.5])



X, t = generate_noise_data(20)
X_train = np.copy(X)
t_train = t[:,None]
N = len(X)
X = np.c_[X, np.ones(X.shape[0])]
a = np.zeros((N,1))
L = np.zeros((N,1))

count = 0
PHI = kernel(X, X)
for _ in range(CountMax):
    for i in range(N):
        a[i] += LR * (1. - t_train[i] * np.dot(PHI, t_train * a)[i])
        if (a[i] < 0):
            a[i] = 0
        elif (a[i] > C):
            a[i] = C
    count += 1

# for _ in range(CountMax):
#     del_L = 1. - t_train * np.dot(PHI, t_train * a)
#     a += LR * del_L
#     a[a<0] = 0
#     a[a>C] = C




S = list(np.where(a.ravel() > 1e-5)[0])
w = (a[S] * t_train[S] * X[S]).sum(axis=0)
b = w[2]



scatter(*X_train.T, c=np.where(t > 0, 'r', 'b'), marker='x', s=50, linewidth=1.5, zorder=3)



scatter(*X[S][:,:2].T, s=80, c='y', marker='o')


Xt,Yt = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
x_test = np.array([Xt, Yt]).reshape(2, -1).T
Z = np.sum(a.ravel() * t * kernel(x_test, X_train), axis=-1) + b
plt.contourf(Xt, Yt, Z.reshape(Xt.shape), alpha=0.4, cmap='jet', levels=np.linspace(min(Z), max(Z), 4))


xlim(-6, 6)
ylim(-6, 6)
show()
