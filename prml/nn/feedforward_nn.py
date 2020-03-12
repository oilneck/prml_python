import numpy as np
from .linear_nn import Linear_NeuralNet
from .setting_layer import *
class Feed_Forward(Linear_NeuralNet):

    def __init__(self,NUM_INPUT:int=1,NUM_HIDDEN:int=3,NUM_OUTPUT:int=1):
        super().__init__(NUM_INPUT,NUM_HIDDEN,NUM_OUTPUT)
        self.xlist = None
        self.tlist = None

    def gradE(self,w):
        self.setW(w)
        E = 0
        gradE = np.zeros(self.getW().size)
        for n in range(len(self.xlist)):
            del_1,del_2 = np.array(self.Back_propagation(self.xlist[n],self.tlist[n,:]))
            gradw1 = np.outer(del_1,np.insert(self.xlist[n],0,1))
            gradw2 = np.outer(del_2,self.layer1.output)
            E += 0.5 * np.dot(del_2,del_2)
            gradE += np.hstack((gradw1.ravel(),gradw2.ravel()))
        return gradE,E

    def setW(self,w):
        self.w1 = w[0:self.n_input * self.n_hidden].reshape(self.n_hidden,self.n_input)
        self.w2 = w[self.n_input*self.n_hidden:].reshape(self.n_output,self.n_hidden)

    def getW(self):
        return np.hstack((self.w1.ravel(),self.w2.ravel()))
