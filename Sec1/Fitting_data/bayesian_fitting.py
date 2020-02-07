import numpy as np
import matplotlib.pyplot as plt
from base_module.poly_feature import Poly_Feature
from fitting.bayesian_regression import Bayesian_Regression

M = 8
noise_NUM = 10


def func(x):
    return np.sin(2 * np.pi * x)

def generate_noise_data(func,noise_NUM,std_dev):
    x_n = np.linspace(0,1,noise_NUM)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n,t_n

# Create the training data
train_x,train_y = generate_noise_data(func,noise_NUM,0.2)

# Create the test data
test_x = np.arange(0, 1.01, 0.01)
test_y = func(test_x)


'''----------------Bayesian Regression------------------------'''
model = Bayesian_Regression(degree=M,alpha=5*10**(-3),beta=11.1)
model.fit(train_x,train_y)
y_mean,y_std = model.predict(test_x,get_std=True)


# plot the test data
plt.plot(test_x,test_y,color='limegreen',label="$\sin(2\pi x)$")
plt.plot(test_x,y_mean,color='red',label="$m(x)$")
plt.fill_between(test_x, y_mean + y_std, y_mean - y_std, facecolor='pink',alpha=0.4,label="std.")

# plot the training data
plt.scatter(train_x,train_y,facecolor="none", edgecolor="b",label="noise",s=50,linewidth=1.5)

# config for drawing
plt.annotate("M={}".format(M), xy=(0.1, -1),fontsize=15)
plt.legend(fontsize=12)
plt.xlabel("x",fontsize=15)
plt.xlim(-0.05, 1.05)
plt.ylim(-1.7, 1.7)
plt.xticks([0,1])
plt.yticks([-1,0,1])
plt.show()
