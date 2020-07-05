import numpy as np

class Metropolis(object):


    def __init__(self, target, prop, dim:int=1):
        '''
        target : collable -> target function [~p(z)]

        prop : random variable object -> proposal distribution [q(z)]
        '''
        self.target = target
        self.prop = prop
        self.dim = dim
        self.path = {'accept':[], 'reject':[]}


    def _check_attr(self):
        cls_name = self.prop.__class__.__name__
        message = "class {} has no attribute [draw] method ".format(cls_name)
        assert hasattr(self.prop, "draw"), message


    def accept_rate(self, x_new, x_old):
        return min(1., self.target(x_new) / self.target(x_old))


    def rvs(self, size, downsample:int=10, init_x:float=1):
        self._check_attr()
        sample = []
        x = np.ones(self.dim) * init_x

        for n in range(size * downsample):
            x_tmp = x + self.prop.draw(sample_size=self.dim)

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
