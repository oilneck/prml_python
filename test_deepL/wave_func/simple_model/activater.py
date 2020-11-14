import numpy as np

class Sigmoid():

    def __init__(self):
        self.func = lambda x: 0.5 + 0.5 * np.tanh(0.5 * x)

    def forward(self, x:np.ndarray, deriv:bool=False):

        output = self.func(x)

        if deriv:
            output = output * (1 - output)

        return output


class Tanh():

    def __init__(self):
        pass

    def forward(self, x:np.ndarray, deriv:bool=False):

        output = np.tanh(x)

        if deriv:
            output = 1 - output ** 2

        return output
