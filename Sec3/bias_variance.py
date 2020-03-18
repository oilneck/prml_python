import numpy as np
import matplotlib.pyplot as plt
from fitting.bayesian_regression import Bayesian_Regression
from base_module import *

def func(x):
    return np.sin(2 * np.pi * x)

def create_noise_data(function,sample_size, std, domain=[0, 1]):
    x = np.linspace(domain[0], domain[1], sample_size)
    np.random.shuffle(x)
    t = function(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


# test data
test_x = np.linspace(0, 1, 200)
test_y = func(test_x)
feature = Gaussian_Feature(np.linspace(0, 1, 24), 0.1)
X_test = feature.transform(test_x)



plt.close()
for i,lamda in enumerate([1e+2, np.e**(0.1), np.e**(-30)]):
    fig = plt.figure(figsize=(8, 3.))
    ax = fig.add_subplot(1,2,1)
    predict_list = []
    for n in range(100):
        train_x,train_y = create_noise_data(func,25,0.25)
        X_train = feature.transform(train_x)
        model = Bayesian_Regression(alpha=lamda,beta=1.)
        model.fit(X_train,train_y)
        y_mean = model.predict(X_test)
        if n < 20:
            ax.plot(test_x,y_mean,color='magenta',linewidth=0.3)
            ax.set_xlabel("x")
        ax.set_ylim([-1.5,1.5])
        plt.xticks([0,1])
        plt.yticks([-1,0,1])
        ax.annotate(r"$\log \,\lambda={:.2g}$".format(np.log(lamda)), xy=(0.5, 0.7),fontsize=12)
        predict_list.append(y_mean)
    ax = fig.add_subplot(1,2,2)
    ax.plot(test_x,test_y,color='lime')
    ax.plot(test_x,np.array(predict_list).mean(axis=0),color='red',linestyle='dashed')
    ax.set_xlabel("x")
    ax.set_ylim([-1.5,1.5])
    plt.xticks([0,1])
    plt.yticks([-1,0,1])
plt.tight_layout()
plt.show()
