import numpy as np
import scipy.optimize

class RejectionSampling(object):


    def __init__(self, target, prop, offset:float=None):
        '''
        target : collable -> target function [~p(z)]

        prop : random variable object -> proposal distribution [q(z)]

        offset : float -> offset constant
        '''
        self.target = target
        self.prop = prop
        self.k = self.fetch_upperLim(offset)



    def fetch_upperLim(self, upper):
        if upper is None:
            xopt = scipy.optimize.fmin(lambda x : -self.target(x), 0, disp=0)
            f_max = self.target(xopt)[0]
            xopt = scipy.optimize.fmin(lambda x: -self.prop.pdf(x), 0, disp=0)
            q_max = self.prop.pdf(xopt)[0]
            max_const = f_max / q_max
        else:
            assert isinstance(upper, (int, float, np.number)),\
            "upper limit must be float or integer number"
            max_const = upper
        return max_const


    def _check_attr(self):
        cls_name = self.prop.__class__.__name__
        message = "class {} has no attribute [draw] method ".format(cls_name)
        assert hasattr(self.prop, "draw"), message


    def rvs(self, size:int=1):
        self._check_attr()
        samples = []
        while len(samples) < size:
            sample = self.prop.draw(sample_size = 1)
            u0 = np.random.uniform(0, self.k * self.prop.pdf(sample))
            if u0 < self.target(sample):
                samples.append(sample[0])
        return np.asarray(samples)
