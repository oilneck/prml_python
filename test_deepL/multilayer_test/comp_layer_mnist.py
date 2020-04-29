import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from deepL_module.nn.multi_layer_nn import Neural_net
from deepL_module.nn.optimizers import *
from deepL_module.nn.cost_functions import *
from deepL_module.base import *

# === loading data ===
(X_train, train_t), (X_test, test_t) = load_mnist(normalize=True)

train_size = X_train.shape[0]
batch_size = 256
max_iter = 1000



# === layers ===
layers = ['tanh', 'relu', 'softplus', 'softsign', 'swish']



model = {}
optimizer = {}
train_loss = {}
train_acc = {}
for key in layers:
    model[key] = Neural_net(n_input=784,
                            n_hidden=[5, 5, 5],
                            n_output=10,
                            alpha=0.01)
    model[key].add([key] * 3)
    model[key].set_loss('categorical_crossentropy')
    optimizer[key] = Adam(lr = 0.01)
    train_loss[key] = []
    train_acc[key] = []


#=== learning ===
for i in range(max_iter):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = X_train[batch_mask]
    t_batch = to_categorical(train_t[batch_mask], cls_num=10)

    for key in layers:
        grads = model[key].gradient(x_batch, t_batch)
        optimizer[key].update(model[key].params, grads)

        # get loss data
        loss = model[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

        # get accuracy data
        acc = model[key].accuracy(x_batch, t_batch)
        train_acc[key].append(acc)


# plot accuracy data
fig = plt.figure(figsize=(18,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
markers = ["o", "v", "s", "*", "D"]

x = np.arange(max_iter)
for n,key in enumerate(layers):
    ax1.plot(x, smooth_filt(train_loss[key]), marker=markers[n],
            markersize=9, markevery=100, label=key, zorder=n, alpha=1 - 0.1 * n)
    ax1.set_xlabel("iterations", fontsize=20)
    ax1.set_title("loss", fontsize=20)
    ax1.set_xlim(-30, max_iter)
    ax1.tick_params(labelsize=12)

    ax2.plot(x, smooth_filt(train_acc[key]), marker=markers[n],
            markersize=9, markevery=100, label=key, zorder=n, alpha=1 - 0.1 * n)
    ax2.set_xlabel("iterations", fontsize=20)
    ax2.set_title("accuracy", fontsize=20)
    ax2.set_xlim(-30, max_iter)
    ax2.tick_params(labelsize=12)


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.tight_layout()
plt.show()
