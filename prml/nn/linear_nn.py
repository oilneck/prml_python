import numpy as np
from nn.regression.neural_network import Neural_Network
class Linear_NeuralNet(Neural_Network):

    def __init__(self,NUM_INPUT:int=1,NUM_HIDDEN:int=3,NUM_OUTPUT:int=1):
        super().__init__(NUM_INPUT,NUM_HIDDEN,NUM_OUTPUT)
        self.add(layer=['tanh','identity'])

    def get_hidden_output(self,test_x):
        test_z = np.zeros((len(test_x),self.n_hidden))
        for n in range(len(test_x)):
            self.Forward_propagation(test_x[n])
            test_z[n] = self.layer1.output
        return test_z
