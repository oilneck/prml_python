import numpy as np
import matplotlib.pyplot as plt
from kernel import *

def generate_noise_data(n_sample):
    cov = [[0.5,0.2], [0.2,0.5]]
    N = int(n_sample // 2)
    cls1 = np.random.multivariate_normal([1.,1.], cov, N).T
    cls2 = np.random.multivariate_normal([-1.,-1.], cov, N).T
    t = np.hstack((np.ones(cls1.shape[1]),np.zeros(cls2.shape[1])))
    return cls1, cls2, t

cls1, cls2, t_train = generate_noise_data(50)
x_train = np.c_[cls1,cls2].T

X,Y = np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
x_test = np.array([X.ravel(), Y.ravel()]).reshape(2,-1).T


model = GP_classifier(kernel=GaussianKernel(1.,7.))
model.fit(x_train, t_train)
Z = model.predict(x_test)

# plotting test data
plt.figure(figsize=(8,4))
plt.scatter(cls1[0],cls1[1],c='r',marker='x',label="class1",s=50,linewidth=1.5)
plt.scatter(cls2[0],cls2[1],facecolor="none", edgecolor="b",label="class2",s=50,linewidth=1.5)

# drawing prediction area
plt.contourf(X,Y, Z.reshape(X.shape), alpha=0.2, levels=np.linspace(min(Z), max(Z), 4),cmap='jet')
plt.colorbar()
plt.xlim(-4.2,4.2)
plt.ylim(-4.2,4.2)
plt.xticks(np.arange(-4,5,2))
plt.yticks(np.arange(-4,5,2))
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
