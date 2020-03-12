import numpy as np
from .feedforward_nn import Feed_Forward
class Scaled_CG(object):

    def __init__(self,NUM_INPUT:int=1,NUM_HIDDEN:int=3,NUM_OUTPUT:int=1):
        self.nn = Feed_Forward(NUM_INPUT,NUM_HIDDEN,NUM_OUTPUT)
        self.sigma0 = 1e-4
        self.w = self.nn.getW()

    def __call__(self,x:np.array):
        return np.vectorize(self.nn.Forward_propagation)(x)

    def set_train_data(self,x:np.array,t:np.array):
        self.nn.xlist = x
        self.nn.tlist = t

    def fit(self,train_x:np.array,train_y:np.array,n_iter:int=500,n_reset:int=50):
        self.set_train_data(train_x,train_y)
        lamda,lamda_bar =1,0
        nablaE,E = self.nn.gradE(self.w)
        origin_E = E
        origin_nablaE = nablaE
        p,r = -nablaE,-nablaE
        success = True
        for k in range(1,n_iter):
            if success:
                sigma = self.sigma0 / np.sqrt(np.dot(p,p))
                delE,E = self.nn.gradE(self.w + sigma * p)
                s = (delE - origin_nablaE) / sigma
                delta = np.dot(p,s)
            delta = delta + (lamda-lamda_bar) * np.dot(p,p)
            if delta < 0:
                lamda_bar = 2 * (lamda - delta/np.dot(p,p))
                delta = -delta + lamda * np.dot(p,p)
                lamda = lamda_bar
            mu = np.dot(p,r)
            alpha = mu / delta
            delE,E = self.nn.gradE(self.w + alpha * p)
            DELTA = 2 * delta * (origin_E - E) / mu**2
            if DELTA > 0:
                origin_E,origin_nablaE,origin_r = E,delE,r
                self.w = self.w + alpha * p
                r = -self.nn.gradE(self.w)[0]
                lamda_bar,success = 0,True
                k += 1
                if divmod(k,n_reset)[1] == 0:
                    p = r
                else:
                    beta = (np.dot(r,r) - np.dot(r,origin_r))/mu
                    p = r + beta * p
                if DELTA >= 0.75:
                    lamda = lamda / 4
                else:
                    lamda_bar,success = lamda,False
            if DELTA < 0.25:
                lamda = lamda + delta*(1-DELTA)/np.dot(p,p)
            if np.linalg.norm(r)==0:
                return
        self.nn.setW(self.w)
