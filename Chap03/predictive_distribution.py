import numpy as np
import matplotlib.pyplot as plt
from base_module import Gaussian_Feature
from fitting.bayesian_regression import Bayesian_Regression

M = 9


def func(x):
    return np.sin(2 * np.pi * x)

def generate_noise_data(func,noise_NUM,std_dev):
    x_n = np.linspace(0,1,noise_NUM)
    np.random.shuffle(x_n)
    t_n = func(x_n) + np.random.normal(scale=std_dev,size=noise_NUM)
    return x_n,t_n

# Create the training data
train_x,train_y = generate_noise_data(func,25,0.2)
feature = Gaussian_Feature(np.linspace(0, 1, M), 0.1)
X_train = feature.transform(train_x)

# Create the test data
test_x = np.arange(0, 1.01, 0.01)
X_test = feature.transform(test_x)
test_y = func(test_x)


'''----Bayesian Regression----'''
model = Bayesian_Regression(alpha=5e-3,beta=2.)


fig = plt.figure(figsize=(8, 6))
for i,noise_num in enumerate([1,2,4,25],1):
    model.fit(X_train[0:noise_num,:],train_y[0:noise_num])
    y_mean,y_std = model.predict(X_test,get_std=True)
    ax = fig.add_subplot(2,2,i)
    plt.scatter(train_x[0:noise_num],train_y[0:noise_num],facecolor="none", edgecolor="b",label="noise",s=50,linewidth=1.5,zorder=3)
    plt.plot(test_x,test_y,color='limegreen',label="$\sin(2\pi x)$",zorder=1)
    plt.plot(test_x,y_mean,color='red',label="mean",zorder=2)
    plt.fill_between(test_x, y_mean + y_std, y_mean - y_std, facecolor='pink',alpha=0.4,label="std.")
    ax.text(0.5,-2.1,r"$x$")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.7, 1.7)
    plt.xticks([0,1])
    plt.yticks([-1,0,1])

# config for drawing
plt.subplots_adjust(right=0.75)
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left', borderaxespad=0, fontsize=12)
plt.tight_layout()
plt.get_current_fig_manager().window.setGeometry(0,15,800,490)
plt.show()
