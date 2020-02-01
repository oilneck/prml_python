import numpy as np
from setting_layer import Linear_Layer, Tanh_Layer
import copy
class Linear_NeuralNet(object):
    def __init__(self,NUM_INPUT:int=2,NUM_HIDDEN:int=4,NUM_OUTPUT:int=1):
        self.n_input = NUM_INPUT
        self.n_hidden = NUM_HIDDEN
        self.n_output = NUM_OUTPUT
        self.testw1 = np.random.random((NUM_HIDDEN,NUM_INPUT))
        self.testw2 = np.random.random((NUM_OUTPUT,NUM_HIDDEN))
        self.layer1 = Tanh_Layer()
        self.layer2 = Linear_Layer()

    def Forward_propagation(self,x):
        input_vec_x = np.insert([x],0,1)
        z = self.layer1.fp(self.testw1,input_vec_x)
        return self.layer2.fp(self.testw2,z)

    def Back_propagation(self,x,t):
        delta_2 = self.Forward_propagation(x)-t
        delta_1 = self.layer1.activation_derivative() * (self.testw2.T @ delta_2)
        return delta_1,delta_2

    def fit(self,xlist,tlist,n_iter:int=1000,learning_rate:float=0.1):
        for loop in range(n_iter):
            for n in range(len(xlist)):
                del_1,del_2 = np.array(self.Back_propagation(xlist[n],tlist[n,:]))
                self.testw1 -= learning_rate * (del_1.reshape(self.n_hidden,1) @ np.insert(xlist[n],0,1).reshape(1,self.n_input))
                self.testw2 -= learning_rate * (del_2.reshape(self.n_output,1) @ self.layer1.output.reshape(1,self.n_hidden))
