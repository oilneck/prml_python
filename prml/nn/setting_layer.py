import numpy as np

class Linear_Layer(object):

    def fp(self,W,x):
        self.output = np.dot(W,x)
        return self.output
    def activation_derivative(self):
        return 1

class Tanh_Layer(object):

    def fp(self,W,x):
        self.output = np.tanh(np.dot(W,x))
        return self.output
    def activation_derivative(self):
        return 1-self.output**2

class Sigmoid_Layer(object):

    def fp(self,W,x):
        self.output = 0.5 + 0.5 * np.tanh(0.5 * np.dot(W,x))
        return self.output
    def activation_derivative(self):
        return self.output * (1 - self.output)