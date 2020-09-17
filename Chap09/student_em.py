import numpy as np
import matplotlib.pyplot as plt
from pd import *

x_train = np.random.normal(size=20)
x_train = np.concatenate([x_train, np.random.normal(loc=20., size=3)])
x = np.linspace(-10, 25, 500)

students = Students_t()
students.fit(x_train)

gaussian = Gaussian()
gaussian.fit(x_train)


plt.hist(x_train, bins=40, color='b', density=True, rwidth=0.7, ec='k', lw=1., label="samples")
plt.plot(x, gaussian.pdf(x), label="gaussian", linewidth=2, color='r')
plt.plot(x, students.pdf(x), label="student's t", lw=2, color='limegreen')
plt.show()
