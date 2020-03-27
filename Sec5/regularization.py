import numpy as np
import matplotlib.pyplot as plt
from nn.feedforward_nn import Feed_Forward

def create_noise_data(Noise_NUM=10):
    x = np.linspace(0, 1, Noise_NUM)[:, None]
    return x, np.sin(2 * np.pi * x) + np.random.normal(scale=0.25, size=(10, 1))

# create training data
train_x,train_y = create_noise_data()

# create test data
test_x = np.linspace(0,1,100)

plt.figure(figsize=(10, 2.5))
for i,(n_unit,hyper_param) in enumerate(zip([1,3,30,30],[0,0,0,0.01]),1):
    plt.subplot(1,4,i)
    model = Feed_Forward(1,n_unit,1,alpha=hyper_param)
    model.compile(optimizer = 'scg')
    model.fit(train_x,train_y)
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
