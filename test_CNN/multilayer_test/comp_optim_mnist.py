import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from deepL_module.nn.multi_layer_nn import Neural_net
from deepL_module.nn.optimizers import *
from deepL_module.nn.cost_functions import *
from deepL_module.base.functions import *

# === loading data ===
(X_train, train_t), (X_test, test_t) = load_mnist(normalize=True)

train_size = X_train.shape[0]
batch_size = 256
max_iter = 2000


# === config for optimizer ===
optimizers = {}
optimizers['SGD'] = SGD(lr=0.1)
optimizers['Momentum'] = Momentum(lr=0.02, momentum=0.9)
optimizers['Adagrad'] = Adagrad(lr=0.0001)
optimizers['RMSprop'] = RMSprop(lr=0.002, rho=0.9)
optimizers['Adam'] = Adam(lr=0.01)


model = {}
train_loss = {}
for key in optimizers.keys():
    model[key] = Neural_net(n_input=784,
                            n_hidden=[100, 100, 100],
                            n_output=10,
                            alpha=0.01,
                            weight_std=0.01)
    model[key].add(['sigmoid', 'sigmoid', 'sigmoid', 'linear'])
    model[key].set_loss('categorical_crossentropy')
    train_loss[key] = []


# === learning ===
for i in range(max_iter):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = X_train[batch_mask]
    t_batch = to_categorical(train_t[batch_mask],10)

    for key in optimizers.keys():
        grads = model[key].gradient(x_batch, t_batch)
        optimizers[key].update(model[key].params, grads)

        loss = model[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 200 == 0:
        print( "===========" + "iteration:" + str(i).rjust(3) + "===========")
        for key in optimizers.keys():
            loss = model[key].loss(x_batch, t_batch)
            print('method:' + key + 'loss:{:.5g}'.rjust(28-len(key)).format(loss))


# === drawing loss data ===
markers = {"SGD": "o", "Momentum": "x", "Adagrad": "s", "RMSprop": "*", "Adam": "D"}
fig=plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)
x = np.arange(max_iter)
for n,key in enumerate(optimizers.keys(),1):
    ax.plot(x, train_loss[key] / np.max(train_loss[key]), marker=markers[key], markevery=100, label=key, zorder=n, alpha=1-0.15*n)
plt.xlabel("iterations",fontsize=20)
plt.title("Comparison of optimizer losses",fontsize=20)
plt.xlim([-30,2020])
plt.legend(fontsize=15)
plt.show()
