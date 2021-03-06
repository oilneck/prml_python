import numpy as np
import matplotlib.pyplot as plt
from base_module import *
from classifier import *

def generate_noise_data(N):
    cov = [[1.,0.8], [0.8,1.]]
    cls1 = np.random.multivariate_normal([-0.5,1], cov, int(N))
    cls2 = np.random.multivariate_normal([1,-0.5], cov, int(N))
    t = np.hstack((np.ones(cls1.shape[0]),np.zeros(cls2.shape[0])))
    return cls1, cls2, t


# create training data
cls1, cls2, train_t = generate_noise_data(50)
train_x = np.concatenate([cls1,cls2])
feature = Poly_Feature(1)
X_train = feature.transform(train_x)

# create test data
X,Y = np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
test_x = np.array([X.ravel(), Y.ravel()]).reshape(2,-1).T
X_test = feature.transform(test_x)

'''bayesian logistic regression'''
model = Bayesian_Logistic_Regression(alpha=1e-3)
model.fit(X_train,train_t)
Z = model.predict(X_test)

# plotting training data
plt.scatter(cls1.T[0],cls1.T[1],c='r',marker='x',label="class1",s=30,linewidth=1.5)
plt.scatter(cls2.T[0],cls2.T[1], facecolor="none", edgecolor="b",label="class2",s=50,linewidth=1.5)

# plotting test data
plt.contourf(X,Y, Z.reshape(X.shape), alpha=0.2, levels=np.linspace(0, 1, 5),cmap='jet')
plt.colorbar()
plt.xlim(-4.5,4.5)
plt.ylim(-4.5,4.5)
plt.show()
