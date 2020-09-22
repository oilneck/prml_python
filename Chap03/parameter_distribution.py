import numpy as np
import matplotlib.pyplot as plt
from fitting.bayesian_regression import Bayesian_Regression
from base_module import Poly_Feature
from scipy.stats import multivariate_normal, norm

def generate_noise_data(func, noise_NUM, std_dev):
    x_n = np.linspace(-.9, .9, noise_NUM)
    np.random.shuffle(x_n)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n, t_n

def likelihood(x, t):
    return norm(np.dot(W, x.T), np.sqrt(1 / model.beta)).pdf(t)

# Create the training data
train_x, train_y = generate_noise_data(lambda x: .5 * x - .3, 20, 0.2)
feature = Poly_Feature(1)
X_train = feature.transform(train_x)

# Create the test data
x = np.linspace(-1, 1, 100)
wx, wy = np.meshgrid(x, x)
W = np.array([wx, wy]).reshape(2, -1).T
X_test = feature.transform(x)



'''----Bayesian Regression----'''
model = Bayesian_Regression(alpha=2, beta=25)



for n, (begin, end) in enumerate([[0, 0], [0, 1], [1, 2], [2, 20]],1):
    model.fit(X_train[begin: end], train_y[begin: end])
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(1,3,1)
    ax.scatter(-0.3, 0.5, s=200, marker="+", color='w', zorder=2)
    wz = likelihood(X_train[begin:begin+1], train_y[begin:begin+1])
    ax.contourf(wx, wy, wz.reshape(wx.shape), 30, cmap='jet')
    plt.gca().set_aspect('equal')
    plt.xticks([-1,0,1])
    plt.yticks([-1,0,1])
    plt.xlabel("$w_0$", fontsize=15)
    plt.ylabel("$w_1$", fontsize=15)
    if n==1:
        ax.set_title("likelihood", fontsize=15)


    ax = fig.add_subplot(132)
    ax.scatter(-0.3, 0.5, s=200, marker="+", zorder=2, color='w')
    wz = multivariate_normal(mean=model.w_mean, cov=model.w_cov).pdf(W)
    ax.contourf(wx, wy, wz.reshape(wx.shape), levels=30, cmap='jet')
    plt.gca().set_aspect('equal')
    plt.xticks([-1,0,1])
    plt.yticks([-1,0,1])
    plt.xlabel("$w_0$", fontsize=15)
    plt.ylabel("$w_1$", fontsize=15)
    if n==1:
        plt.title("prior/posterior", fontsize=15)



    ax = fig.add_subplot(133)
    ax.scatter(train_x[:end], train_y[:end], s=50, facecolor="none", edgecolor="b", lw=1, zorder=3)
    ax.plot(x, model.posterior(X_test, n_sample=6), c="r", lw=2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.xticks([-1,0,1])
    plt.yticks([-1,0,1])
    plt.gca().set_aspect('equal', adjustable='box')
    if n==1:
        plt.title("data space", fontsize=15)
    plt.tight_layout()
    plt.show()
