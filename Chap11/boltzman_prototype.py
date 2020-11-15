import numpy as np
import matplotlib.pyplot as plt
import copy

def create_train(num:int=100, batch_size:int=50):
    x = np.ones((num, n_unit))
    inv_rowidx = np.random.choice(x.shape[0], batch_size)
    x[inv_rowidx] *= -1
    inv_colidx = np.random.choice(x.shape[1], n_unit - 1)
    x[inv_colidx] *= -1
    return x

def calc_grad(x_train:np.ndarray):
    train_size = x_train.shape[0]
    term1 = x_train.T @ x_train / train_size
    tmp = np.tanh( params['field'] + params['weight'].dot(x_train.T) )
    term2 = np.dot(tmp, x_train) / train_size
    dW = term1 - term2
    db = np.mean(x_train - tmp.T, axis=0)
    return {'weight': -dW, 'field':-db.reshape(-1, 1)}


def update(params:dict, grads:dict, lr:float=1):
    for key in params.keys():
        params[key] -= lr * grads[key]

def fit(X_train:np.ndarray, max_iter:int=100, lr:float=1):

    errors = []
    for n in range(int(max_iter)):
        err = calc_energy(X_train)
        errors.append(err)
        update(params, calc_grad(X_train), lr=lr)
        if np.allclose(err, calc_energy(X_train)):
            print('num_iter', n)
            break

    return errors

def calc_energy(X:np.ndarray):
    train_size = X.shape[0]
    term1 = np.sum(X.T * (params['weight'] @ X.T), axis=0)
    term2 = (params['field'].T @ x_train.T).ravel()
    term3 = np.log( np.cosh(params['weight'] @ X.T + params['field']) )
    err = -(term1 + term2 - term3.sum(axis=0)).sum() / train_size
    return err


n_unit = 3
x_train = create_train()

params = {}
params['weight'] = np.random.random((n_unit, n_unit))
params['field'] = np.random.random((n_unit, 1))

err = fit(x_train, max_iter=200)


plt.plot(np.arange(len(err)), err, c='limegreen', label='normal cost func.')
plt.title('energy', fontsize=15)
plt.legend(fontsize=15)
plt.show()
