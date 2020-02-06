"""This is a test program."""
import numpy as np
import matplotlib.pyplot as plt
from fitting.multiple_regression import Multiple_Regression
M = 9
noise_NUM = 10

def func(x):
    return np.sin(2 * np.pi * x)
#Generating Noise data
def generate_noise_data(func,noise_NUM,std_dev):
    #noise_NUM:sample size, std_dev:standard deviation
    x_n = np.linspace(0,1,noise_NUM)#linspace(start,stop,Deivision number)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n,t_n

x_n,t_n = generate_noise_data(func,noise_NUM,0.3)
test_x = np.arange(0, 1.01, 0.01)
model = Multiple_Regression(M,lamda=(5*10**(-3))/11.1)
model.fit(x_n,t_n)
test_y = model.predict(test_x)

plt.plot(test_x,func(test_x),color='lime',label="$\sin(2\pi x)$")
plt.plot(test_x,test_y,color='red',label="Fitting")
plt.scatter(x_n,t_n,facecolor="none", edgecolor="b",label="noise")
plt.legend()
plt.show()
