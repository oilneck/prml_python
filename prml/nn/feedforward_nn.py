import numpy as np
from nn.regression.neural_network import Neural_Network


class Feed_Forward(Neural_Network):

    def __init__(self,NUM_INPUT:int=1,NUM_HIDDEN:int=3,NUM_OUTPUT:int=1,alpha:float=0):
        super().__init__(NUM_INPUT,NUM_HIDDEN,NUM_OUTPUT)
        self.add()
        self.xlist = None
        self.tlist = None
        self.optimizer = None
        self.hyper_param = alpha #Regularization


    def __call__(self,x:np.array):
        return np.vectorize(self.Forward_propagation)(x)


    def gradE(self,w):
        self.setW(w)
        E = 0
        gradE = np.zeros(self.getW().size)
        for n in range(len(self.xlist)):
            del_1,del_2 = np.array(self.Back_propagation(self.xlist[n],self.tlist[n,:]))
            gradw1 = np.outer(del_1,np.insert(self.xlist[n],0,1))
            gradw2 = np.outer(del_2,self.layer1.output)
            E += 0.5 * np.dot(del_2,del_2)
            gradE += np.hstack((gradw1.ravel(),gradw2.ravel()))
        return gradE + self.hyper_param * w, E + 0.5 * self.hyper_param * np.dot(w,w)

    def compile(self,optimizer:str='scg'):
        from nn.optimizer import Scaled_CG,Adam,SGD
        if optimizer == 'scg':
            self.optim_routine = Scaled_CG(self.n_input-1,self.n_hidden,self.n_output,self.hyper_param)
        elif optimizer == 'adam':
            self.optim_routine = Adam(self.n_input-1,self.n_hidden,self.n_output)
        elif optimizer == 'sgd':
            self.optim_routine = SGD(self.n_input-1,self.n_hidden,self.n_output,self.hyper_param)
        else:
            self.optim_routine = SGD(self.n_input-1,self.n_hidden,self.n_output,self.hyper_param)

    def fit(self,train_x:np.ndarray,train_y:np.ndarray,**param):
        self.w = self.getW()
        self.setW(self.optim_routine.update(train_x,train_y,self.w,**param))


    def setW(self,w):
        self.w1 = w[0:self.n_input * self.n_hidden].reshape(self.n_hidden,self.n_input)
        self.w2 = w[self.n_input*self.n_hidden:].reshape(self.n_output,self.n_hidden)

    def getW(self):
        return np.hstack((self.w1.ravel(),self.w2.ravel()))
