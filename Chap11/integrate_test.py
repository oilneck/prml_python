import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from pd import *
from sampling import *

x = np.linspace(-5, 5, 100)
target = lambda x: np.reciprocal(1 + x ** 2) # ~p
normalize_factor = integrate.quad(target, -np.inf, np.inf)[0] # Zp
func = lambda x: np.exp(- x ** 4)

int_val = integrate.quad(lambda x:target(x) * func(x), -np.inf, np.inf)[0]
print('correct value:=', int_val)

sampler = Metropolis(target=target, prop=Gaussian(0, 1))
samples = sampler.rvs(int(1e+4), downsample=1, init_x=0).ravel()

'''Approximate integral with Monte Carlo method'''
expval = func(samples).sum() / len(samples)
print('expected value:=', expval * normalize_factor)

fig = plt.figure(figsize=(7,4))
fig.add_subplot(111)
plt.plot(x, func(x), c='b')
plt.show()
