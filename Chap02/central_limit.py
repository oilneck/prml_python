import numpy as np
import matplotlib.pyplot as plt
from pd.uniform import Uniform
np.random.seed()

prob = Uniform(low=0, high=1)
plt.figure(figsize=(10, 3))
size = 10000


for i,N in enumerate([1,2,10],1):
    plt.subplot(1, 3, i)
    plt.xlim(0, 1)
    plt.xticks([0,0.5,1])
    plt.ylim(0, 5)
    plt.annotate("N={}".format(N), (0.1, 4.))
    sample = np.zeros(size)
    for _ in range(N):
        sample += prob.draw(size)
    plt.hist(sample/N, bins=20,color='b',density=True,rwidth = 0.7,edgecolor='black',lw=1.)
plt.show()
