import numpy as np
from functools import reduce

class Node(object):

    def __init__(self, ndim:int=2, alpha:float=10., beta:float=5.):
        '''
        QUBO system
        s = {+1, -1}
        x = (s + 1) / 2 = {0, +1}
        '''
        self.neighbors = []
        self.messages = {}
        self.prob = None

        self.ndim = ndim
        self.alpha = alpha
        self.beta = beta


    def add_neighbor(self, node):
        self.neighbors.append(node)


    def get_neighbor(self):
        return self.neighbors


    def init_messages(self):
        for neighbor in self.neighbors:
            self.messages[neighbor] = np.ones(self.ndim)


    def send_message(self, node):
        message_list = list(self.messages.values())
        inflow_message = np.prod(message_list, axis=0) / self.messages[node]
        transfer_matrix = np.exp(-self.beta * (-np.eye(self.ndim) + 1))
        message = np.dot(transfer_matrix, inflow_message)
        node.messages[self] = message / message.sum()


    def marginalize(self):
        message_list = list(self.messages.values())
        prob = np.prod(message_list, axis=0)
        self.prob = prob / prob.sum()


    def likelihood(self, value):
        assert value in [0, 1], 'observed value must be 0 or 1'

        observed_value = np.array([int(value), int(not value)])
        message = np.exp(-self.alpha * observed_value)

        self.messages[self] = message
