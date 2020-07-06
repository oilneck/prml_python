import numpy as np
from sampling.metropolis import Metropolis

class MetropolisHastings(Metropolis):


    def __init__(self, target, prop, dim:int=1):
        super().__init__(target, prop, dim)


    def accept_rate(self, x_tmp, x):
        frac = self.target(x_tmp) * self.prop.pdf(x - x_tmp)
        nume = self.target(x) * self.prop.pdf(x_tmp - x)
        return min(1, frac / nume)


    def rvs(self, size, downsample:int=10, init_x:float=1):
        self._check_attr()
        sample = []
        x = np.ones(self.dim) * init_x

        for n in range(size * downsample):
            x_tmp = x + self.prop.draw(1).ravel()

            if np.random.uniform() < self.accept_rate(x_tmp, x):
                self.path['accept'].append([x_tmp, x])
                x = x_tmp
            else:
                self.path['reject'].append([x_tmp, x])

            if n % downsample == 0:
                sample.append(x)

        self.path['accept'] = np.asarray(self.path['accept'])
        self.path['reject'] = np.asarray(self.path['reject'])
        return np.asarray(sample)
