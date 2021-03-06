"""This is a test program for confirmation the RMSE."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

F_size = 15

M_list = range(0,10)
noise_Num = len(M_list)
sigma = 0.2
RMSE_train = []
RMSE_test = []

def rmse(a,b):
    return np.sqrt(np.mean(np.square(a - b)))


def func(x):
    return np.sin(2 * np.pi * x)

def generate_noise_data(func,noise_NUM,std_dev):
    x_n = np.linspace(0,1,noise_NUM)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n,t_n

train_x,train_t = generate_noise_data(func,noise_Num,sigma)
train_data = pd.DataFrame(np.array([train_x,train_t]).transpose(),columns=['x','t'])
test_x,test_t   = generate_noise_data(func,100,sigma)
test_data = pd.DataFrame(np.array([test_x,test_t]).transpose(),columns=['x','t'])

for M in M_list:
    weight_list = np.polyfit(train_data.x.values,train_data.t.values,M)
    RMSE_train.append(rmse(np.polyval(weight_list,np.array(train_data.x)),train_data.t))
    RMSE_test.append(rmse(np.polyval(weight_list,np.array(test_data.x)),test_data.t))



plt.plot(M_list,RMSE_train,label='Training data',color='blue')
plt.plot(M_list,RMSE_train,color='none',marker='o',markeredgecolor='blue',markersize=8,linewidth=0)
plt.plot(M_list,RMSE_test,label='Test data',color='red')
plt.plot(M_list,RMSE_test,color='none',marker='o',markeredgecolor='red',markersize=8,linewidth=0)
plt.yticks( [0,0.5,1] )
plt.xticks( [0,3,6,9] )
plt.xlabel("M",fontsize=F_size)
plt.ylabel(r"$E_{\mathrm{RMSE}}$",fontsize=F_size)
plt.legend(loc='upper center',fontsize=F_size)
plt.show()
