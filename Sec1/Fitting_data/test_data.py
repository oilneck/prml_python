"""This is a test program."""
import numpy as np
import matplotlib.pyplot as plt
noise_NUM = 10
def func(x):
    return np.sin(2 * np.pi * x)
#Generating Noise data
def generate_noise_data(func,noise_NUM,std_dev):
    #noise_NUM:sample size, std_dev:standard deviation
    x_n = np.linspace(0,1,noise_NUM)#linspace(start,stop,Deivision number)
    t_n = func(x_n)+ np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n,t_n


x = np.arange(0, 1.01, 0.01)
y = func(x)
x_n,t_n = generate_noise_data(func,noise_NUM,0.3)
plt.plot(x,y,color='lime',label="$\sin(2\pi x)$")
plt.scatter(x_n,t_n,marker='o',color='blue',label="noise")
plt.hlines(y=t_n[3], xmin=-2,xmax=x_n[3], linewidth=1,linestyle='--',color = 'k')
plt.vlines(x=x_n[3], ymin=-2, ymax=t_n[3], linewidth=1,linestyle='--',color = 'k')

plt.text(-0.1,t_n[3], "$t_n$",fontsize=15)
plt.text(x_n[3]-0.01,-1.7, "$x_n$",fontsize=15)
plt.yticks( [-1.5, 0.0, 1.5] )
plt.xticks( [0.0, 0.5, 1.0] )
plt.xlim(-0.05, 1.05)
plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()
