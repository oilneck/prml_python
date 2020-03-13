import numpy as np
from nn.regression.neural_network import Neural_Network
class Classifier_NeuralNet(Neural_Network):

    def __init__(self,NUM_INPUT:int=2,NUM_HIDDEN:int=4,NUM_OUTPUT:int=1):
        super().__init__(NUM_INPUT,NUM_HIDDEN,NUM_OUTPUT)
        self.add(layer=['tanh','sigmoid'])

    def predict(self,test_PHI:np.ndarray):
        output = np.array( [self.Forward_propagation(test_PHI[n,:]) for n in range(len(test_PHI))] )
        return output
