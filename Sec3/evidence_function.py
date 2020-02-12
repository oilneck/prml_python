import numpy as np
import matplotlib.pyplot as plt
from fitting.optimal_bayesian_regression import Optimal_Bayesian_Regression

Font_size = 15
noise_NUM = 50
max_M = 8

def func(x):
    return np.sin(2*np.pi*x)

#Generating Noise data
def generate_noise_data(func,noise_NUM,std_dev):
    x_n = np.linspace(0,1,noise_NUM)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n,t_n



train_x,train_y = generate_noise_data(func,noise_NUM,0.2)
evidence_List = []
model_List = []

for m in range(max_M+1):
    model = Optimal_Bayesian_Regression(degree=m,alpha=100,beta=100)
    model.fit(train_x,train_y)
    evidence_List.append(model.evidence_function(train_x,train_y))
    model_List.append(model)

deg_index = np.argmax(evidence_List)
optimal_model = model_List[deg_index]
test_x = np.linspace(0,1,100)
y_mean,y_std = optimal_model.predict(test_x,get_std=True)

fig = plt.figure(figsize=(11,4.0))

# plot log evidence
plt.subplot(1,2,1)
plt.grid()
plt.subplots_adjust(right=0.8)
plt.plot(np.arange(max_M+1),np.array(evidence_List))
plt.xlabel("M",fontsize=Font_size)
plt.ylabel("evidence",fontsize=Font_size)

# plot test data
plt.subplot(1,2,2)
plt.plot(test_x,func(test_x),color='limegreen',label="$\sin(2\pi x)$")
plt.plot(test_x,y_mean,color='red',label="$m(x)$")
plt.fill_between(test_x,y_mean+y_std,y_mean-y_std,facecolor='pink',alpha=0.4,label="std.")
plt.scatter(train_x,train_y,facecolor="none",edgecolor="b",label="noise",s=50,linewidth=1.5)
plt.legend(bbox_to_anchor=(1.05,0.5),loc='upper left',borderaxespad=0,fontsize=Font_size)
plt.annotate("M={}".format(deg_index),xy=(0.7,1),fontsize=Font_size)
plt.annotate(r"$\alpha$={:.2g}".format(optimal_model.alpha),xy=(0.,-1.2),fontsize=13)
plt.annotate(r"$\beta$={:.3g}".format(optimal_model.beta),xy=(0.,-1.5),fontsize=13)
plt.tight_layout()
plt.xlabel("x",fontsize=15)
plt.xlim(-0.07,1.02)
plt.ylim(-1.7,1.7)
plt.xticks([0,1])
plt.yticks([-1,0,1])
plt.show()
