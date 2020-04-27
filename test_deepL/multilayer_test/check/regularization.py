import numpy as np
import matplotlib.pyplot as plt
from deepL_module.nn.multi_layer_nn import *

def create_noise_data(Noise_NUM=10):
    x = np.linspace(0, 1, Noise_NUM)[:, None]
    return x, np.sin(2 * np.pi * x) + np.random.normal(scale=0.25, size=(10, 1))

# create training data
train_x,train_y = create_noise_data()

# create test data
test_x = np.linspace(0,1,100)[:,None]

plt.figure(figsize=(12, 3))
for i,(n_unit,hyper_param) in enumerate(zip([1,3,50,50],[0,0,0,1e-2]),1):
    plt.subplot(1,4,i)
    model = Neural_net(1, n_unit, 1, alpha=hyper_param)
    model.add(['sigmoid', 'linear'])
    routine = Adam(lr=0.1)
    model.compile(loss='sum_squared_error', optimizer=routine)
    model.fit(train_x,train_y,n_iter=1000)
    test_y = model(test_x)
    plt.plot(test_x,test_y,color="r",zorder=1)
    plt.scatter(train_x.ravel(), train_y.ravel(), marker="x", color="b",zorder=2,s=30)
    plt.annotate("M={}".format(n_unit), (0.7, 0.5),fontsize=10)
    plt.xticks([0,1])
    plt.yticks([-1,0,1])
    if not(np.allclose(0,hyper_param)):
    	plt.title('Regularized',color='forestgreen')
    plt.subplots_adjust(wspace=1.5)
plt.tight_layout()
plt.show()
