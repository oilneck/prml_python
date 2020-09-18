import numpy as np
import matplotlib.pyplot as plt
from pd import *

samples = np.random.normal(size=30)
outlier = np.concatenate([samples, np.random.normal(loc=20., size=5)])
x = np.linspace(-10, 25, 500)

students = Students_t()
gaussian = Gaussian()


fig = plt.figure(figsize=(12,5))
for n, x_train in enumerate([samples, outlier]):
    ax = fig.add_subplot(1,2,n+1)
    plt.hist(x_train, bins=40,
             density=True,
             rwidth=.7, ec='k',
             lw=.9, label="samples",
             color='#7F7FCC', range=(-10,25))
    students.fit(x_train)
    gaussian.fit(x_train)
    plt.plot(x, gaussian.pdf(x), label="gaussian", linewidth=2, color='limegreen')
    plt.plot(x, students.pdf(x), label="student's t", lw=2, color='r')
    plt.xlim([-10, 25])
plt.legend(fontsize=15)
plt.show()
