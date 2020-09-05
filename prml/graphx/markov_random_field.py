import numpy as np

class MarkovRandomField(object):

    def __init__(self):
        self.__nodes = []
        self.label = {}

    @property
    def nodes(self):
        pass

    @nodes.getter
    def nodes(self):
        return self.__nodes


    def add_node(self, name, node):
        self.__nodes.append(node)
        self.label[name] = node

    def get_node(self, name):
        return self.label[name]

    def clear_messages(self):
        for node in self.nodes:
            node.init_messages()

    def calc_proba(self):
        for node in self.nodes:
            node.marginalize()

    def message_passing(self, n_iter=20):

        self.clear_messages()

        for _ in range(n_iter):
            for node in self.nodes:
                for neighbor in node.get_neighbor():
                    node.send_message(neighbor)

        self.calc_proba()
