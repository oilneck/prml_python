"""This is a test program."""
import numpy as np
import matplotlib.pyplot as plt
from prml.fitting.linear_regression import Linear_Regression
M = 9
noise_NUM = 10
#lamda = np.e**(-18)
lamda = (5*10**(-3))/11.1
def func(x):
    return np.sin(2 * np.pi * x)
#Generating Noise data
def generate_noise_data(func,noise_NUM,std_dev):
    #noise_NUM:sample size, std_dev:standard deviation
    x_n = np.linspace(0,1,noise_NUM)#linspace(start,stop,Deivision number)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n,t_n

x = np.arange(0, 1.01, 0.01)
y = func(x)
x_n,t_n = generate_noise_data(func,noise_NUM,0.3)

model = Linear_Regression(M,lamda=lamda)
model.fit(x_n,t_n)
Fit_y = np.poly1d(model.weight_vector)
plt.plot(x,y,color='lime',label="$\sin(2\pi x)$")
plt.plot(x,Fit_y(x),color='red',label="Fitting")
plt.scatter(x_n,t_n,facecolor="none", edgecolor="b",label="noise")
plt.legend()
plt.show()
