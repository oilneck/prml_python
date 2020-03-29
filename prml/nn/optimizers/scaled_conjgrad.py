import numpy as np
from nn.feedforward_nn import Feed_Forward
class Scaled_CG(object):

    def __init__(self,n_in,n_hid,n_out,regularization_coe):
        self.nn = Feed_Forward(n_in,n_hid,n_out,regularization_coe)


    def set_param(self,param):
        if 'n_reset' in param.keys():
            self.n_reset = param['n_reset']
        else:
            self.n_reset = 100

        if 'sigma_0' in param.keys():
            self.sigma0 = param['sigma_0']
        else:
            self.sigma0 = 1e-4

        if 'n_iter' in param.keys():
            self.n_iter = param['n_iter']
        else:
            self.n_iter = 1000


    def set_train_data(self,x:np.array,t:np.array):
        self.nn.xlist = x
        self.nn.tlist = t


    def update(self,w_vec,**kwargs):
        self.set_param(kwargs)
        lamda,lamda_bar = 1 , 0
        nablaE,E = self.nn.gradE(w_vec)
        origin_E = E
        origin_nablaE = nablaE
        p,r = -nablaE,-nablaE
        success = True
        for k in range(1,self.n_iter):
            if success:
                sigma = self.sigma0 / np.sqrt(np.dot(p,p))
                delE,E = self.nn.gradE(w_vec + sigma * p)
                s = (delE - origin_nablaE) / sigma
                delta = np.dot(p,s)
            delta = delta + (lamda-lamda_bar) * np.dot(p,p)
            if delta < 0:
                lamda_bar = 2 * (lamda - delta/np.dot(p,p))
                delta = -delta + lamda * np.dot(p,p)
                lamda = lamda_bar
            mu = np.dot(p,r)
            alpha = mu / delta
            delE,E = self.nn.gradE(w_vec + alpha * p)
            DELTA = 2 * delta * (origin_E - E) / mu**2
            if DELTA > 0:
                origin_E,origin_nablaE,origin_r = E,delE,r
                w_vec = w_vec + alpha * p
                r = -self.nn.gradE(w_vec)[0]
                lamda_bar,success = 0,True
                k += 1
                if divmod(k,self.n_reset)[1] == 0:
                    p = r
                else:
                    beta = (np.dot(r,r) - np.dot(r,origin_r))/mu
                    p = r + beta * p
                if DELTA >= 0.75:
                    lamda = np.clip(lamda / 4,1e-15,1e50)
                else:
                    lamda_bar,success = lamda,False
            if DELTA < 0.25:
                lamda = np.clip(lamda + delta*(1-DELTA)/np.dot(p,p),1e-15,1e50)
            if np.linalg.norm(r)==0:
                return
        return(w_vec)
