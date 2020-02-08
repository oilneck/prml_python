"""This is a test program."""
import numpy as np
import matplotlib.pyplot as plt
Font_size = 15
M = 10
noise_NUM = 30
alpha = 5 * 10 ** (-3)
beta = 11
def func(x):
    return np.sin(2 * np.pi* x)
#Generating Noise data
def generate_noise_data(func,noise_NUM,std_dev):
    #noise_NUM:sample size, std_dev:standard deviation
    x_n = np.linspace(0,1,noise_NUM)#linspace(start,stop,Deivision number)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n,t_n

def make_basis(test_x):#input :one factor of vector
    return np.array([test_x]*(M+1))**np.arange(0,M+1)

def make_design_matrix(x_n,t_n,M):
    PHI_T = np.array([x_n]*(M+1))**(np.array([np.arange(0,M+1)]*(len(t_n))).T)
    S_inv = beta*np.dot(PHI_T,PHI_T.T) + alpha*np.eye(M+1,M+1)
    S_N = np.linalg.inv(S_inv)
    Desired_T =  np.dot(PHI_T,t_n)
    m_N = beta * np.dot(S_N,Desired_T)
    return PHI_T,S_N,m_N

def make_Average_Variance(test_x):
    PHI_T,S,m_N = make_design_matrix(train_x,train_t,M)
    basis = make_basis(test_x)
    m_x = np.dot(basis.T,m_N)
    s_x = np.sqrt(beta**(-1) + np.dot(basis.T,np.dot(S,basis)))
    return m_x,s_x

def evidence_function(M):
    PHI_T,S_N,m_N = make_design_matrix(train_x,train_t,M)
    Error_Value = (beta/2) * (np.linalg.norm(train_t-np.dot(PHI_T.T,m_N), ord=2))**2 + (alpha/2) * np.dot(m_N.T,m_N)
    return (noise_NUM * np.log(beta) + M * np.log(alpha)  +  np.linalg.slogdet(S_N)[1])/2 - Error_Value




x = np.arange(-0.001, 1.01, 0.01)
y = func(x)
train_x,train_t = generate_noise_data(func,noise_NUM,0.2)
M_list = np.arange(M+1)
evidence_List = []

for i, degree in enumerate(M_list):
    evidence_List.append(evidence_function(degree))


fig = plt.figure(figsize=(11, 4.0))
#plot log evidence
plt.subplot(1, 2, 1)
plt.subplots_adjust(right=0.8)
plt.plot(M_list,np.array(evidence_List))
plt.xlabel("M",fontsize=Font_size)
plt.ylabel("evidence",fontsize=Font_size)
#plot bayesian fitting
make_Ave_Var = np.vectorize(make_Average_Variance)
m_x,s_x = make_Ave_Var(x)
plt.subplot(1,2,2)
plt.plot(x,y,color='limegreen',label="$\sin(2\pi x)$")
plt.plot(x,m_x,color='red',label="$m(x)$")
plt.fill_between(x,m_x+s_x,m_x-s_x,facecolor='pink',alpha=0.4,label="std.")
plt.scatter(train_x,train_t,facecolor="none", edgecolor="b",label="noise",s=50,linewidth=1.5)
plt.annotate("M={}".format(M), xy=(0.1, -1),fontsize=Font_size)
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left', borderaxespad=0, fontsize=10)
plt.xlabel("x",fontsize=15)
plt.xlim(-0.05, 1.05)
plt.ylim(-1.7, 1.7)
plt.xticks([0,1])
plt.yticks([-1,0,1])
plt.show()
