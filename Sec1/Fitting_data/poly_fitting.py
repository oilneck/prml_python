import numpy as np
import matplotlib.pyplot as plt
from fitting.multiple_regression import Multiple_Regression
F_size = 15
noise_NUM = 10
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
M_list = [0,1,3,10]
x_n,t_n = generate_noise_data(func,noise_NUM,0.3)
for i, deg in enumerate(M_list,0):
    plt.subplot(2, 2, i+1)
    model = Multiple_Regression(degree=deg)
    model.fit(x_n,t_n)
    plt.plot(x,y,color='lime',label="$\sin(2\pi x)$")
    plt.plot(x,model.predict(x),color='red',label="Fitting")
    plt.scatter(x_n,t_n,facecolor="none", edgecolor="b",label="noise")
    plt.text(0.7,0.8,"M={}".format(deg),fontsize=F_size)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.5, 1.5)
plt.subplots_adjust(right=0.8)
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left', borderaxespad=0, fontsize=10)
plt.show()
