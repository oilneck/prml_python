import numpy as np
import matplotlib.pyplot as plt
from sampling.boltzmann_machine import BoltzmannMachine



def create_train(n_unit:int, num:int=100, batch_size:int=50):
    x = np.ones( (int(num), int(n_unit)) )
    inv_rowidx = np.random.choice(x.shape[0], batch_size)
    x[inv_rowidx] *= -1
    return x

n_unit = 5
X_train = create_train(n_unit, num=100)
model = BoltzmannMachine(n_unit)
err = model.fit(X_train, n_iter=200)


plt.plot(np.arange(len(err)), err, c='limegreen', label='L2-regularization')
plt.title('energy', fontsize=15)
plt.legend(fontsize=15)
plt.show()
