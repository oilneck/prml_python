import numpy as np
import matplotlib.pyplot as plt
from pd import *

prob = Binomial(10,0.25)

for m in np.arange(0,11,1):
    plt.bar(m,prob.pdf(m),color='b',edgecolor='k')
plt.xlabel("m")
plt.xticks(np.arange(0,11,1))
plt.yticks(np.arange(0,0.4,0.1))
plt.show()
