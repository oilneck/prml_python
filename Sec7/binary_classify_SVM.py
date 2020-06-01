import numpy as np
from pylab import *
from kernel import *

LR = 0.05
C = 1#np.inf
CountMax = 100

def generate_noise_data(N):
    correlation= [0.9, 0.5]
    cov = [correlation, np.flip(correlation)]
    N = int(N // 2)
    cls1 = np.random.multivariate_normal([-0.5,1], cov, int(N)).T
    cls2 = np.random.multivariate_normal([1,-0.5], cov, int(N)).T
    t = np.hstack((np.ones(cls1.shape[1]), -np.ones(cls2.shape[1])))
    return np.c_[cls1, cls2].T, t


def kernel(x, y):
    args = -0.5 * 0.5 * np.sum((x - y) ** 2, axis=-1)
    return np.exp(args) + 0.5

def dL(i):
    ans = 0
    for j in range(0,N):
        ans += L[j] * t[i] * t[j] * kernel(X[i], X[j])
    return (1 - ans)


X, t = generate_noise_data(20)
X_train = np.copy(X)
N = len(X)
X = np.c_[X, np.ones(X.shape[0])]
L = np.zeros((N,1))

count = 0
while (count < CountMax):
    for i in range(N):
        L[i] = L[i] + LR * dL(i)    # ラグランジュ乗数の更新
        if (L[i] < 0):
            L[i] = 0
        elif (L[i] > C):
            L[i] = C
    count += 1


S = list(np.where(L.ravel() > 1e-5)[0])



t_train = t[:,None]
w = (L[S] * t_train[S] * X[S]).sum(axis=0)
b = w[2]



scatter(*X_train.T, c=np.where(t > 0, 'r', 'b'), marker='x', s=50, linewidth=1.5, zorder=3)



scatter(*X[S][:,:2].T, s=80, c='y', marker='o')


kernel = GaussianKernel(*[1.,0.5,0.5])
Xt,Yt = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
x_test = np.array([Xt, Yt]).reshape(2, -1).T
Z = np.sum(L.ravel() * t * kernel(x_test, X_train), axis=-1) + b
plt.contourf(Xt, Yt, Z.reshape(Xt.shape), alpha=0.4, cmap='jet', levels=np.linspace(min(Z), max(Z), 4))


xlim(-6, 6)
ylim(-6, 6)
show()
