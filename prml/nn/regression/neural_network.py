import numpy as np
from .setting_layer import Linear_Layer, Tanh_Layer, Sigmoid_Layer, ReLU_Layer
class Neural_Network(object):

    def __init__(self,NUM_INPUT:int=1,NUM_HIDDEN:int=3,NUM_OUTPUT:int=1):
        self.n_input = NUM_INPUT + 1
        self.n_hidden = NUM_HIDDEN
        self.n_output = NUM_OUTPUT
        self.w1 = np.random.random((self.n_hidden,self.n_input))
        self.w2 = np.random.random((self.n_output,self.n_hidden))
        self.layer1 = None
        self.layer2 = None

    def __call__(self,test_x:np.array):
        return np.vectorize(self.Forward_propagation)(test_x)

    def add(self,layer:list=['tanh','identity']):
        activation = []
        for name in layer:
            if name == 'identity':
                activation.append(Linear_Layer())
            elif name == 'tanh':
                activation.append(Tanh_Layer())
            elif name == 'sigmoid':
                activation.append(Sigmoid_Layer())
            elif name == 'relu':
                activation.append(ReLU_Layer())
        self.layer1 = activation[0]
        self.layer2 = activation[1]

    def Forward_propagation(self,x):
        input_vec_x = np.insert([x],0,1)
        z = self.layer1.fp(self.w1,input_vec_x)
        return self.layer2.fp(self.w2,z)

    def Back_propagation(self,x,t):
        delta_2 = self.Forward_propagation(x)-t
        delta_1 = self.layer1.activation_derivative() * (self.w2.T @ delta_2)
        return delta_1,delta_2

    def fit(self,xlist,tlist,n_iter:int=1000,learning_rate:float=0.1):
        for loop in range(n_iter):
            for n in range(len(xlist)):
                del_1,del_2 = np.array(self.Back_propagation(xlist[n],tlist[n,:]))
                self.w1 -= learning_rate * np.outer(del_1,np.insert(xlist[n],0,1))
                self.w2 -= learning_rate * np.outer(del_2,self.layer1.output)
