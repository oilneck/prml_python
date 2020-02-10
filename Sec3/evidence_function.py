import numpy as np
import matplotlib.pyplot as plt
from fitting.bayesian_regression import Bayesian_Regression
from base_module.poly_feature import Poly_Feature

Font_size = 15
noise_NUM = 30
max_M = 8

def func(x):
    return np.sin(2 * np.pi* x)

#Generating Noise data
def generate_noise_data(func,noise_NUM,std_dev):
    x_n = np.linspace(0,1,noise_NUM)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n,t_n


def evidence_function(M):
    model = Bayesian_Regression(M,1,0.6)
    model.fit(train_x,train_y)
    m_N = model.w_mean
    S_N = model.w_cov
    beta = model.beta
    alpha = model.alpha
    feature = Poly_Feature(M)
    PHI = feature.transform(train_x)
    Error_Value = (beta/2) * np.square(train_y - PHI @ m_N).sum() + (alpha/2) * np.sum(m_N**2)
    return 0.5 * (len(train_y) * np.log(beta) + M * np.log(alpha)  +  np.linalg.slogdet(S_N)[1]) - Error_Value




train_x,train_y = generate_noise_data(func,noise_NUM,0.2)
evidence_List = []

for degree in range(max_M+1):
    evidence_List.append(evidence_function(degree))



#plot log evidence
plt.plot(np.arange(max_M+1),np.array(evidence_List))
plt.xlabel("M",fontsize=Font_size)
plt.ylabel("evidence",fontsize=Font_size)
plt.show()
