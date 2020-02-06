"""This is a test program."""
import numpy as np
import matplotlib.pyplot as plt
from base_module.poly_feature import Poly_Feature

M = 9
noise_NUM = 10
alpha = 5 * 10 ** (-3)
beta =11.1
def func(x):
    return np.sin(2 * np.pi * x)
#Generating Noise data
def generate_noise_data(func,noise_NUM,std_dev):
    #noise_NUM:sample size, std_dev:standard deviation
    x_n = np.linspace(0,1,noise_NUM)#linspace(start,stop,Deivision number)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n,t_n

def make_basis(test_x):
    return np.array([test_x]*(M+1))**np.arange(0,M+1)

def make_design_matrix(x_n,t_n):
    feature = Poly_Feature(degree=M)
    PHI_T = feature.transform(x_n).T
    S_inv = beta*np.dot(PHI_T,PHI_T.T) + alpha*np.eye(M+1,M+1)
    Desired_T =  np.dot(PHI_T,t_n)
    return PHI_T,np.linalg.inv(S_inv),Desired_T

def make_Average_Variance(test_x):
    PHI_T,S,Desired_T = make_design_matrix(x_n,t_n)
    basis = make_basis(test_x)
    m_x = beta * np.dot(basis,np.dot(S,Desired_T))
    s_x = np.sqrt(beta**(-1) + np.dot(basis.T,np.dot(S,basis)))
    return m_x,s_x



x = np.arange(0, 1.01, 0.01)
y = func(x)
x_n,t_n = generate_noise_data(func,noise_NUM,0.2)
m_x,s_x = np.vectorize(make_Average_Variance)(x)




plt.plot(x,y,color='limegreen',label="$\sin(2\pi x)$")
plt.plot(x,m_x,color='red',label="$m(x)$")
plt.fill_between(x,m_x+s_x,m_x-s_x,facecolor='pink',alpha=0.4,label="std.")
plt.scatter(x_n,t_n,facecolor="none", edgecolor="b",label="noise",s=50,linewidth=1.5)
plt.annotate("M={}".format(M), xy=(0.1, -1),fontsize=15)
plt.legend(fontsize=12)
plt.xlabel("x",fontsize=15)
plt.xlim(-0.05, 1.05)
plt.ylim(-1.7, 1.7)
plt.xticks([0,1])
plt.yticks([-1,0,1])
plt.show()
