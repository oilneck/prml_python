import numpy as np

class SASampling(object):


    def __init__(self, beta:float=10):
        '''
        Hamiltonian
        H = sum_{ij} J_{ij} s_i.s_j + sum_i h_i.s_i

        -----------------------------
        J_{ij} : Exchange interaction
        h_i : External magnetic field
        '''
        self.beta = beta
        self.h = None
        self.J = None
        self.indices = []
        self.sample = {}
        self.energy = None
        self.states = []
        self.energies = []


    def init_state(self):

        idx_list = []
        for (site1, site2) in self.J.keys():
            idx_list.append(site1)
            idx_list.append(site2)

        self.indices = list(set(idx_list))
        self.indices.sort()

        for idx in self.indices:
            self.sample[idx] = np.random.choice([1, -1])


    def get_energy(self, state:dict):

        state = state.copy()
        E = 0
        for (site1, site2), val in self.J.items():
            E += state[site1] * state[site2] * val

        for site, val in self.h.items():
            E += state[site] * val

        return E


    def accept_rate(self, x_tmp, x):
        frac = np.exp(-self.beta * self.get_energy(x_tmp)).clip(max=1e+100)
        nume = np.exp(-self.beta * self.get_energy(x)).clip(max=1e+100)
        return min(1, frac / nume)


    def sample_ising(self, h:dict, J:dict, n_iter:int=10):
        self.h = h
        self.J = J
        self.init_state()

        for _ in range(n_iter):
            new_state = self.sample.copy()
            idx = np.random.choice(self.indices)
            new_state[idx] *= -1

            if np.random.uniform() < self.accept_rate(new_state, self.sample):
                self.sample = new_state

            self.states.append(list(self.sample.values()))
            self.energy = self.get_energy(self.sample)
            self.energies.append(self.energy)
