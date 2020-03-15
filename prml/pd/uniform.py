import numpy as np

class Uniform(object):

    def __init__(self,low:float=0.,high:float=1.):
        self.low = low
        self.high = high
        self.value = 1 / (self.high - self.low)

    def draw(self,sample_size:int=1000):
        return np.random.uniform(self.low,self.high,sample_size)

    def pdf(self,x):
        return np.where((x > self.low) & (x < self.high), self.value, 0)
