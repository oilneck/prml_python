import numpy as np
import matplotlib.pyplot as plt
from base_module.sigmoid_feature import Sigmoid_Feature

F_size = 15
feature = Sigmoid_Feature(mean=[0.],std=1.)
x = np.arange(-10, 10, 0.01)
y = feature.transform(x)[:,1]
plt.close()
plt.axvline(x=0, ymin=0, ymax=1, linewidth=1,linestyle='--',color = 'k')
plt.axhline(y=0.5,xmin=-10,xmax=10,linewidth=1,linestyle='--',color = 'k')
plt.plot(x, y,color='r')
plt.xlim(-7,7)
plt.ylim(-0.1,1.1)
plt.xticks([-5,0,5],fontsize=F_size)
plt.yticks([0,0.5,1],fontsize=F_size)
plt.show()
