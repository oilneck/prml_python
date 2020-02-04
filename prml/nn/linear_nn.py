import numpy as np
from .setting_layer import Linear_Layer, Tanh_Layer
class Linear_NeuralNet(object):
    def __init__(self,NUM_INPUT:int=2,NUM_HIDDEN:int=4,NUM_OUTPUT:int=1):
        self.n_input = NUM_INPUT
        self.n_hidden = NUM_HIDDEN
        self.n_output = NUM_OUTPUT
        self.w1 = np.random.random((self.n_hidden,self.n_input))
        self.w2 = np.random.random((self.n_output,self.n_hidden))
        self.layer1 = Tanh_Layer()
        self.layer2 = Linear_Layer()

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
                self.w1 -= learning_rate * (del_1.reshape(self.n_hidden,1) @ np.insert(xlist[n],0,1).reshape(1,self.n_input))
                self.w2 -= learning_rate * (del_2.reshape(self.n_output,1) @ self.layer1.output.reshape(1,self.n_hidden))

    def get_hidden_output(self,test_x):
        test_z = np.zeros((len(test_x),self.n_hidden))
        for n in range(len(test_x)):
            self.Forward_propagation(test_x[n])
            test_z[n] = self.layer1.output

        return np.flip(test_z, axis=1)[:,0:-1]
