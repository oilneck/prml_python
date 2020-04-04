import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from deepL_module.nn.two_layer_net import Two_layer_net
from deepL_module.nn.optimizers import *
from deepL_module.nn.cost_functions import *
from deepL_module.base.functions import *

# === loading data ===
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# === config for optimizer ===
optimizers = {}
optimizers['SGD'] = SGD(lr = 0.01)
optimizers['Momentum'] = Momentum()
optimizers['Adagrad'] = Adagrad()
optimizers['Adam'] = Adam(lr = 0.3)


model = {}
train_loss = {}
for key in optimizers.keys():
    model[key] = Two_layer_net(784,50,10)
    model[key].add(['relu','linear'])
    model[key].set_loss('categorical_crossentropy')
    train_loss[key] = []


# === learning ===
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = to_categorical(t_train[batch_mask],10)

    for key in optimizers.keys():
        grads = model[key].gradient(x_batch, t_batch)
        optimizers[key].update(model[key].params, grads)

        loss = model[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = model[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# === drawing loss data ===
markers = {"SGD": "o", "Momentum": "x", "Adagrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    norm = np.max(train_loss[key])
    plt.plot(x, train_loss[key] / norm, marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations",fontsize=20)
plt.ylabel("loss",fontsize=20)
plt.ylim(0.15, 1.)
plt.legend(fontsize=20)
plt.show()
