import numpy as np
from scipy import integrate
from scipy.misc import derivative
import copy

class PredictWaveFunction(object):

    def __init__(self, n_hidden):
        # Simple NeuralNetwork 1-n_hidden-1

        self.n_hidden = n_hidden
        self.params = {}
        self.params['W1'] = np.random.random((self.n_hidden, 1))
        self.params['W2'] = np.random.random((1, self.n_hidden))
        self.params['b1'] = np.ones((self.n_hidden, 1))
        self.x_range = [-5, 5]
        self.info = {'energies':[]}

    def act_func(self, activation:np.ndarray, deriv:bool=False) -> np.ndarray:

        output = np.tanh(activation)

        if deriv:
            output = 1 - output ** 2

        return output


    def phi(self, x:np.ndarray) -> np.ndarray:

        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x = x[None,:]

        activation = np.dot(self.params['W1'], x).reshape(self.n_hidden, -1)
        activation += self.params['b1']
        z = self.act_func(activation)
        a_output = np.dot(self.params['W2'], z)
        return np.exp(a_output).ravel()


    def get_average(self, func) -> float:
        f = lambda x: self.phi(x) * func(x) * self.phi(x)
        return integrate.quad(f, *self.x_range)[0] / self.get_norm()


    def get_norm(self) -> float:
        return integrate.quad(lambda x:self.phi(x) ** 2, *self.x_range)[0]


    def hamiltonian_operate(self, func) -> float:
        H_phi = lambda x: x ** 2 * self.phi(x) - derivative(self.phi, x, dx=1e-3, n=2)
        f = lambda x: self.phi(x) * func(x) * H_phi(x)
        val = integrate.quad(f, *self.x_range)[0]
        return val / self.get_norm()


    def calc_energy(self) -> float:
        return self.hamiltonian_operate(lambda x:1)


    def get_grad(self) -> np.ndarray:

        E_H = self.calc_energy()

        params = self.get_params()
        bl_params = params[self.n_hidden:2*self.n_hidden]
        bias = params[2*self.n_hidden:]

        grads = []
        for n, val in enumerate(params[:self.n_hidden]):
            s = lambda x: bl_params[n] * x * self.act_func(val * x + bias[n], deriv=True)
            E_sH = self.hamiltonian_operate(s)
            E_s = self.get_average(s)
            grads.append(E_sH - E_s * E_H)

        for n, val in enumerate(params[:self.n_hidden]):
            s = lambda x: self.act_func(val * x + bias[n], deriv=False)
            E_sH = self.hamiltonian_operate(s)
            E_s = self.get_average(s)
            grads.append(E_sH - E_s * E_H)

        # update bias param
        for n, val in enumerate(params[:self.n_hidden]):
            s = lambda x: bl_params[n] * self.act_func(val * x + bias[n], deriv=True)
            E_sH = self.hamiltonian_operate(s)
            E_s = self.get_average(s)
            grads.append(E_sH - E_s * E_H)

        return np.array(grads)



    def update(self, n_iter:int=100, alpha:float=.7):

        for n in range(int(n_iter)):
            #print('n:', n, ' E=', self.hamiltonian_operate(lambda x:1))
            self.info['energies'].append(self.calc_energy())
            w_old = self.get_params()
            w_new = w_old - alpha * self.get_grad()
            self.set_params(w_new)
            if np.allclose(w_new, w_old):
                print('max_iter', n)
                break


    def get_params(self) -> np.ndarray:
        W1_vect = copy.copy(self.params['W1']).ravel()
        W2_vect = copy.copy(self.params['W2']).ravel()
        b1_vect = copy.copy(self.params['b1']).ravel()
        return np.concatenate([W1_vect, W2_vect, b1_vect])


    def set_params(self, vect_params:np.ndarray):
        if isinstance(vect_params, list):
            vect_params = np.asarray(vect_params)
        n_hidden = self.n_hidden
        self.params['W1'] = vect_params[:n_hidden].reshape((n_hidden, 1))
        self.params['W2'] = vect_params[n_hidden:2*n_hidden].reshape((1, n_hidden))
        self.params['b1'] = vect_params[2*n_hidden:].reshape((n_hidden, 1))
