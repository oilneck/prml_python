import numpy as np
from nn.regression.neural_network import Neural_Network


class Feed_Forward(Neural_Network):

    def __init__(self,NUM_INPUT:int=1,NUM_HIDDEN:int=3,NUM_OUTPUT:int=1,alpha:float=0):
        super().__init__(NUM_INPUT,NUM_HIDDEN,NUM_OUTPUT)
        self.add()
        self.xlist = None
        self.tlist = None
        self.hyper_param = alpha #Regularization
        self.init_w = self.getW()


    def __call__(self,x:np.array):
        return np.vectorize(self.Forward_propagation)(x)


    def gradE(self,w):
        self.setW(w)
        [self.xlist,self.tlist] = self.check_dim(self.xlist,self.tlist)
        E = 0
        gradE = np.zeros(self.getW().size)
        for n in range(len(self.xlist)):
            del_1,del_2 = np.array(self.Back_propagation(self.xlist[n],self.tlist[n,:]))
            gradw1 = np.outer(del_1,np.insert(self.xlist[n],0,1))
            gradw2 = np.outer(del_2,self.layer1.output)
            E += 0.5 * np.dot(del_2,del_2)
            gradE += np.hstack((gradw1.ravel(),gradw2.ravel()))
        return gradE + self.hyper_param * w, E + 0.5 * self.hyper_param * np.dot(w,w)

    def optimizer(self,method:str='sgd'):
        unit = [self.n_input-1,self.n_hidden,self.n_output,self.hyper_param]
        from nn.optimizers import Scaled_CG,Adam,SGD,RMSprop,Adagrad,Momentum
        if method == 'scg':
            self.optim_routine = Scaled_CG(*unit)
        elif method == 'adam':
            self.optim_routine = Adam(*unit)
        elif method == 'sgd':
            self.optim_routine = SGD(*unit)
        elif method == 'rmsprop':
            self.optim_routine = RMSprop(*unit)
        elif method == 'adagrad':
            self.optim_routine = Adagrad(*unit)
        elif method == 'momentum':
            self.optim_routine = Momentum(*unit)
        else:
            self.optim_routine = SGD(*unit)

    def fit(self,train_x:np.ndarray,train_y:np.ndarray,**param):
        weight_vect = self.getW()
        self.setW(self.optim_routine.update(train_x,train_y,weight_vect,**param))


    def setW(self,w):
        self.w1 = w[0:self.n_input * self.n_hidden].reshape(self.n_hidden,self.n_input)
        self.w2 = w[self.n_input*self.n_hidden:].reshape(self.n_output,self.n_hidden)

    def getW(self):
        return np.hstack((self.w1.ravel(),self.w2.ravel()))

    def clear(self):
        self.setW(self.init_w)
