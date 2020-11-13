import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from pd import *
from sampling import *

x = np.linspace(0, 1, 100)
target = lambda x: Uniform(0, 1).pdf(x)
func = lambda x: 4 * np.sqrt(1 - x ** 2)

int_val = integrate.quad(func, 0, 1)[0]
print('correct value:=', int_val)

sampler = MetropolisHastings(target=target, prop=Gaussian(0, 1))
samples = sampler.rvs(int(1e+4), downsample=1, init_x=0).ravel()
samples = samples[np.abs(samples) <= 1]
expval = func(samples).sum() / len(samples)
print('expected value:=', expval)

fig = plt.figure(figsize=(7,4))
fig.add_subplot(111)
plt.plot(x, func(x), c='b')
plt.show()
