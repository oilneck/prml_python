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


    def accept_proba(self, x_new, x_old):
        return min(1., self.target(x_new) / self.target(x_old))


    def _check_attr(self):
        cls_name = self.prop.__class__.__name__
        message = "class {} has no attribute [draw] method ".format(cls_name)
        assert hasattr(self.prop, "draw"), message


    def rvs(self, size, downsample=10):
        self._check_attr()
        sample = []
        x = np.zeros(self.dim)
        for n in range(size * downsample):
            x_new = self.prop.draw(sample_size = self.dim)
            x_new += x
            if np.random.uniform() < self.accept_proba(x_new, x):
                x = x_new
            if n % downsample == 0:
                sample.append(x)
        return np.asarray(sample)
