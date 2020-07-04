import numpy as np

class SIR(object):


    def __init__(self, target, prop):
        '''
        target : collable -> target function [~p(z)]

        prop : random variable object -> proposal distribution [q(z)]
        '''
        self.target = target
        self.prop = prop


    def rvs(self, size:int=1):
        samples = self.prop.draw(sample_size = size)
        weights = self.target(samples) / self.prop.pdf(samples)
        weights /= weights.sum()
        return np.random.choice(samples, size, p=weights)
