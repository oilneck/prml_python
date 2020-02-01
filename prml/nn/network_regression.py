import numpy as np

class Linear_NeuralNet(object):
    def __init__(self,NUM_INPUT:int=2,NUM_HIDDEN:int=4,NUM_OUTPUT:int=1):
        self.w1 = np.random.random((NUM_HIDDEN,NUM_INPUT))
        self.w2 = np.random.random((NUM_OUTPUT,NUM_HIDDEN))
