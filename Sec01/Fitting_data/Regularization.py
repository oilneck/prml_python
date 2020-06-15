"""This is a test program."""
import numpy as np
import matplotlib.pyplot as plt
from fitting.multiple_regression import Multiple_Regression
from base_module import Poly_Feature

M = 9
noise_NUM = 10

def func(x):
    return np.sin(2 * np.pi * x)
#Generating Noise data
def generate_noise_data(func,N,std_dev):
    #noise_NUM:sample size, std_dev:standard deviation
    x_n = np.linspace(0,1,N)#linspace(start,stop,Deivision number)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n,t_n

# train data
x_n,t_n = generate_noise_data(func,noise_NUM,0.3)
feature = Poly_Feature(M)
X_train = feature.transform(x_n)

# test data
test_x = np.arange(0, 1.01, 0.01)
X_test = feature.transform(test_x)

'''Polynomial Regression'''
model = Multiple_Regression(alpha=(5*10**(-3))/11.1)
model.fit(X_train,t_n)
test_y = model.predict(X_test)

plt.plot(test_x,func(test_x),color='lime',label="$\sin(2\pi x)$")
plt.plot(test_x,test_y,color='red',label="Fitting")
plt.scatter(x_n,t_n,facecolor="none", edgecolor="b",label="noise")
plt.legend()
plt.show()
