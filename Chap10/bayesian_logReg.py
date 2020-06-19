import numpy as np
import matplotlib.pyplot as plt
from base_module import *
from classifier import VariationalClassifier

n_sample = 50

def make_blobs(n_sample):
    cls1 = np.random.normal(size=(n_sample // 2, 2), loc=(+1.8, 0), scale=.5)
    cls2 = np.random.normal(size=(n_sample // 2, 2), loc=(-1.8, 0), scale=.5)
    X = np.concatenate([cls1, cls2])
    return X, np.where(X[:, 0] > 0, 1, 0)


def plot_train_data(ax):
    N = n_sample // 2
    ax.scatter(*train_x[:N, :].T, marker='o', fc='none', ec='r', lw=1.5, s=70)
    ax.scatter(*train_x[N:, :].T, marker='x', lw=2, s=70, c='b')



# training & test data set
train_x, train_t = make_blobs(n_sample)
feature = Poly_Feature(degree=1)
X_train = feature.transform(train_x)

X, Y = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
X_test = np.array([X, Y]).reshape(2, -1).T
X_test = feature.transform(X_test)


'''Variational logistic regression'''
model = VariationalClassifier()
model.fit(X_train, train_t)
Z = model.predict(X_test)


# variational estimation data
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(121)
plot_train_data(ax)
ax.contour(X, Y, Z.reshape(X.shape), levels=np.linspace(min(Z), max(Z), 8), cmap='jet')
plt.xlim(-4,4)


# plot posterior sample
ax = fig.add_subplot(122)
plot_train_data(ax)
colors = ['orange', 'blue', 'limegreen', 'cyan', 'magenta']
for c in colors:
    prob = model.posterior(X_test)
    ax.contour(X, Y, prob.reshape(X.shape), levels=1, colors=c)
plt.xlim(-4, 4)
plt.show()
