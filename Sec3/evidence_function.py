import numpy as np
import matplotlib.pyplot as plt
from fitting.optimal_bayesian_regression import Optimal_Bayesian_Regression

Font_size = 13
noise_NUM = 30
max_M = 8

def func(x):
    return 50 * np.sin(0.1 * np.pi * x)

#Generating Noise data
def generate_noise_data(func,noise_NUM,std_dev,margin=[0,1]):
    x_n = np.linspace(margin[0],margin[1],noise_NUM)
    np.random.shuffle(x_n)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=x_n.shape)
    return x_n,t_n


#----training data----
train_x,train_y = generate_noise_data(func,noise_NUM,10,[0,20])
evidence_List = []
model_List = []

# parameter estimation
for m in range(max_M+1):
    model = Optimal_Bayesian_Regression(degree=m,alpha=100,beta=100)
    model.fit(train_x,train_y)
    evidence_List.append(model.evidence_function(train_x,train_y))
    model_List.append(model)

#----test data----
deg_index = np.argmax(evidence_List)
optimal_model = model_List[deg_index]
test_x = np.linspace(0,20,100)
y_mean,y_std = optimal_model.predict(test_x,get_std=True)



# plot log evidence
fig = plt.figure(figsize=(9,3.0))
plt.subplot(1,2,1)
plt.grid()
plt.subplots_adjust(right=0.8)
plt.plot(np.arange(max_M+1),np.array(evidence_List))
plt.xlabel("M",fontsize=Font_size)
plt.ylabel("evidence",fontsize=Font_size)

# plot test data
ax=fig.add_subplot(1,2,2)
ax.plot(test_x,func(test_x),color='limegreen',label="$50\sin(\pi x/10)$")
ax.plot(test_x,y_mean,color='red',label="prediction")
ax.fill_between(test_x,y_mean+y_std,y_mean-y_std,facecolor='pink',alpha=0.4,label="std.")
ax.scatter(train_x,train_y,facecolor="none",edgecolor="b",label="noise",s=50,linewidth=1.5)
ax.legend(bbox_to_anchor=(1.05,0.5),loc='upper left',borderaxespad=0,fontsize=Font_size)
ax.text(0.9,0.9,"M={}".format(deg_index),ha='right',va='top',transform=ax.transAxes,fontsize=15)
ax.text(1.65,1,r"$\alpha$={:.1e}".format(optimal_model.alpha),ha='right',va='top',transform=ax.transAxes,fontsize=Font_size)
ax.text(1.65,0.85,r"$\beta$={:.1e}".format(optimal_model.beta),ha='right',va='top',transform=ax.transAxes,fontsize=Font_size)
plt.tight_layout()
plt.xlabel("x",fontsize=15)
plt.show()
