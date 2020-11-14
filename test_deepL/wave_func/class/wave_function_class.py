import numpy as np
from scipy import integrate
from scipy.misc import derivative

class WaveFunction(object):

    def __init__(self):

        self.params = {}
        self.W1 = np.random.random((2, 1))
        self.W2 = np.random.random((1, 2))
        self.b1 = np.ones((2, 1))
        self.x_range = [-5, 5]


    def get_params(self):
        W1_vect = self.W1.ravel()
        W2_vect = self.W2.ravel()
        b1_vect = self.b1.ravel()
        return np.concatenate([W1_vect, W2_vect, b1_vect])


    def set_params(self, vect_params:np.ndarray):

        if isinstance(vect_params, list):
            vect_params = np.asarray(vect_params)

        self.W1 = vect_params[:2].reshape((2, 1))
        self.W2 = vect_params[2:4].reshape((1, 2))
        self.b1 = vect_params[4:].reshape((2, 1))


    def act_func(self, activation:np.ndarray, deriv:bool=False):

        output = np.tanh(activation)
        if deriv:
            output = 1 - output ** 2

        return output


    def phi(self, x:np.ndarray):

        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x = x[None,:]

        activation = np.dot(self.W1, x) + self.b1
        z = self.act_func(activation)
        a_output = np.dot(self.W2, z)

        return np.exp(a_output).ravel()


    def get_average(self, func):
        f = lambda x: self.phi(x) * func(x) * self.phi(x)
        return integrate.quad(f, *self.x_range)[0] / self.get_norm()


    def get_energy(self):
        return self._hamiltonian_operate(lambda x:1)


    def get_norm(self):
        return integrate.quad(lambda x:self.phi(x) ** 2, *self.x_range)[0]


    def update(self, n_iter:int=50, alpha:float=.7):

        for n in range(int(n_iter)):
            w_old = self.get_params()
            w_new = w_old - alpha * self._calc_grad()
            self.set_params(w_new)


    def _hamiltonian_operate(self, func):

        deriv_phi = lambda x: - 0.5 *  derivative(self.phi, x, dx=1e-3, n=2)
        H_phi = lambda x: 0.5 * x ** 2 * self.phi(x) + deriv_phi(x)
        f = lambda x: self.phi(x) * func(x) * H_phi(x)
        value = integrate.quad(f, *self.x_range)[0]
        return value / self.get_norm()


    def _calc_grad(self):

        E_H = self.get_energy()

        w1, w2, w3, w4, b1, b2 = self.get_params()

        s1 = lambda x: w3 * x * self.act_func(w1 * x + b1, deriv=True)
        s2 = lambda x: w4 * x * self.act_func(w2 * x + b2, deriv=True)
        s3 = lambda x: self.act_func(w1 * x + b1)
        s4 = lambda x: self.act_func(w2 * x + b2)
        s5 = lambda x: w3 * self.act_func(w1 * x + b1, deriv=True)
        s6 = lambda x: w4 * self.act_func(w2 * x + b2, deriv=True)

        grads = []
        for theta in [s1, s2, s3, s4, s5, s6]:
            E_sH = self._hamiltonian_operate(theta)
            E_s = self.get_average(theta)
            grad = 2 * (E_sH - E_s * E_H)
            grads.append(grad)

        return np.array(grads)
