import numpy as np
from scipy import integrate
from scipy.misc import derivative
import copy
import os, joblib

class PredictWaveFunction(object):
    '''Simple NeuralNetwork 1-n_hidden-1

    Attributes:
        n_hidden (int): The number of units in hidden layer
        params (dict): model parameter
        x_range (list): Integral interval
        info (dict): Information for wave-func. model
    '''

    def __init__(self, n_hidden:int=2):

        self.n_hidden = n_hidden
        self._init_params()
        self.x_range = [-5, 5]
        self.info = {'energies':[], 'num_hidden':self.n_hidden}


    def _init_params(self):
        self.params = {}
        self.params['W1'] = np.random.random((self.n_hidden, 1))
        self.params['W2'] = np.random.random((1, self.n_hidden))
        self.params['b1'] = np.ones((self.n_hidden, 1)) * 1.5


    def act_func(self, activation:np.ndarray, deriv:bool=False) -> np.ndarray:

        '''1. Activation function

        Args:
            activation (np.ndarray): activation.
            deriv (bool, default=bool): differentiation or not.

        Returns:
            output (np.ndarray): The value activated by the activation function.
        '''

        output = np.tanh(activation)

        if deriv:
            output = 1 - output ** 2

        return output


    def get_average(self, func) -> float:

        '''2. Calculating expectations

        Args:
            func (function):
                The physical quantity for which you want to calculate
                the expected value.

        Returns:
            value (float): Expected value.

        '''
        f = lambda x: self.phi(x) * func(x) * self.phi(x)
        return integrate.quad(f, *self.x_range)[0] / self.get_norm()


    def get_energy(self) -> float:

        '''3. Calculating energy

        Returns:
            value (float): Expectation value of Hamiltonian representing <H>
        '''
        return self._hamiltonian_operate(lambda x:1)


    def get_norm(self) -> float:

        '''4. Calculating normalize factor

        Returns:
            output (float): normalize factor.
        '''
        return integrate.quad(lambda x:self.phi(x) ** 2, *self.x_range)[0]


    def get_params(self) -> np.ndarray:

        '''5. Getter for the model parameter.

        Returns:
            array (np.ndarray): List of model parameter.
        '''
        W1_vect = copy.copy(self.params['W1']).ravel()
        b1_vect = copy.copy(self.params['b1']).ravel()
        W2_vect = copy.copy(self.params['W2']).ravel()
        return np.concatenate([W1_vect, b1_vect, W2_vect])


    def set_params(self, vect_params:np.ndarray):

        '''7. Setter for the model parameter.

        Args:
            vect_params (np.ndarray): 1Dimensionalized Parameters.
        '''

        if isinstance(vect_params, list):
            vect_params = np.asarray(vect_params)

        n_hid = self.n_hidden
        self.params['W1'] = vect_params[:n_hid].reshape((n_hid, 1))
        self.params['b1'] = vect_params[n_hid:2*n_hid].reshape((n_hid, 1))
        self.params['W2'] = vect_params[2*n_hid:].reshape((1, n_hid))



    def phi(self, x:np.ndarray) -> np.ndarray:

        '''6. Wave-function

        Args:
            x (np.ndarray): input array.

        Returns:
            a_output (np.ndarray): output of this neural network.
        '''

        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x = x[None,:]

        activation = np.dot(self.params['W1'], x) + self.params['b1']
        z_output = self.act_func(activation)
        a_output = np.dot(self.params['W2'], z_output)

        return np.exp(a_output).ravel()


    def update(self, n_iter:int=100, alpha:float=.7):

        '''8. Updating model parameter

        Args:
            n_iter (int, default=100): The number of parameter updates.
            alpha (float): Learning rate.
        '''

        for n in range(int(n_iter)):
            self.info['energies'].append(self.get_energy())
            w_old = self.get_params()
            w_new = w_old - alpha * self._calc_grad()
            self.set_params(w_new)

            if np.allclose(w_new, w_old):
                print('max_iter', n)
                break


    def save(self, path:str=None, name:str=None):
        self.info['params'] = self.get_params()
        if path is None and name is None:
            path = os.path.dirname(os.path.abspath(__file__))
            path += '/wf_model.pkl'
        elif path is None and name is not None:
            path = os.path.dirname(os.path.abspath(__file__))
            path += '/' + name + '.pkl'
        elif path is not None and name is None:
            pass
        else:
            raise Exception("Could not specify 'path' and 'name'")

        with open(path, 'wb') as f:
            joblib.dump(self.info, f, compress=3)


    def load(self, path:str=None):
        if path is None:
            path = os.path.dirname(os.path.abspath(__file__))
            path += '/wf_model.pkl'

        with open(path, 'rb') as f:
            info = joblib.load(f)

        self.info = info.copy()
        self.n_hidden = self.info['num_hidden']
        self._init_params()
        self.set_params(self.info['params'])



    def _hamiltonian_operate(self, func) -> float:

        '''Hamiltonian operater

        Args:
            func (function): Functions that invoke the Hamiltonian operator.

        Returns:
            value (float): Expectation value.
        '''
        deriv_phi = lambda x: - derivative(self.phi, x, dx=1e-3, n=2) * 0.5
        H_phi = lambda x: 0.5 * x ** 2 * self.phi(x) + deriv_phi(x)
        f = lambda x: self.phi(x) * func(x) * H_phi(x)
        value = integrate.quad(f, *self.x_range)[0]
        return value / self.get_norm()


    def _calc_grad(self) -> np.ndarray:

        '''Calculating gradient.

        Returns:
            grads (np.ndarray): Gradients of energy with respect to parameters
        '''

        E_H = self.get_energy()
        n_hidden = self.n_hidden

        front, bias, back = self.get_params().reshape(3, n_hidden)
        f_params = np.tile(np.array([front, bias]).T, (3, 1))
        coef = np.concatenate([np.tile(back, 2), np.ones(n_hidden)])
        f_params = np.hstack([f_params, coef.reshape(-1, 1)])

        flags = [True] * 2 * n_hidden + [False] * n_hidden

        grads = []
        for n, ((val1, val2, coe), flag) in enumerate(zip(f_params, flags)):

            f = lambda x: coe * self.act_func(val1 * x + val2, flag)

            if n < n_hidden:
                f = lambda x: coe * self.act_func(val1 * x + val2, flag) * x

            E_s = self.get_average(f)
            E_sH = self._hamiltonian_operate(f)
            grads.append(E_sH - E_s * E_H)

        return np.asarray(grads) * 2
