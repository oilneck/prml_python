"""This is a test program for confirmation the Over fitting."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd

M = 9
noise_Num_list = [15,100]
sigma = 0.2


def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))


def func(x):
    return np.sin(2 * np.pi * x)
#Generating Noise data
def generate_noise_data(func,noise_NUM,std_dev):
    #noise_NUM:sample size, std_dev:standard deviation
    x_n = np.linspace(0,1,noise_NUM)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n,t_n
def generate_test_data(noise_Num):
    test_x,test_t = generate_noise_data(func,noise_Num,sigma)
    test_data = pd.DataFrame(np.array([test_x,test_t]).transpose(),columns=['x','t'])
    return test_data

fig = plt.figure(figsize=(11, 4.0))

x=np.arange(0,1.01,0.01)
for i,Num in enumerate(noise_Num_list,0):
    plt.subplot(1, 2, i+1)
    test_data = generate_test_data(Num)
    weight_list = np.polyfit(test_data.x.values,test_data.t.values,M)
    plt.plot(x,np.polyval(weight_list,x),color='r',label="Fitting")
    plt.plot(x,func(x),color='lime',label="$\sin(2\pi x)$")
    plt.scatter(test_data.x,test_data.t,marker='o',facecolor="none", edgecolor="b",label="noise")
    plt.yticks( [-1,0,1] )
    plt.xticks( [0,1] )
    plt.xlabel("x")
    plt.ylabel("t")
    plt.text(0.8,0.8,"N={}".format(Num))
    plt.xlim([-0.1,1.1])
    plt.ylim([-1.5,1.5])
plt.subplots_adjust(right=0.8)
plt.text(1.2,1,"M={}".format(M))
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left', borderaxespad=0, fontsize=10)
plt.show()
