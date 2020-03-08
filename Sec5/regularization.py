import numpy as np
import matplotlib.pyplot as plt
from nn.scaled_conjgrad import Scaled_CG

def create_noise_data(Noise_NUM=10):
    x = np.linspace(0, 1, Noise_NUM)[:, None]
    return x, np.sin(2 * np.pi * x) + np.random.normal(scale=0.25, size=(10, 1))

# create training data
train_x,train_y = create_noise_data()

# create test data
test_x = np.linspace(0,1,100)

plt.figure(figsize=(9, 3))
for n,M in enumerate([1,3,30]):
    plt.subplot(1,3,n + 1)
    model = Scaled_CG(1,M,1)
    model.set_train_data(train_x,train_y)
    model.fit()
    test_y = model(test_x)
    plt.plot(test_x.ravel(),test_y.ravel(),color="r",zorder=1)
    plt.scatter(train_x.ravel(), train_y.ravel(), marker="x", color="b",zorder=2,s=30)
    plt.annotate("M={}".format(M), (0.7, 0.5),fontsize=10)
    plt.xticks([0,1])
    plt.yticks([-1,0,1])
    plt.subplots_adjust(wspace=1.5)
plt.tight_layout()
plt.show()
