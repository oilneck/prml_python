import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from deepL_module.nn.multi_layer_nn import Neural_net
from deepL_module.nn.optimizers import *
from deepL_module.base import *
import copy

# === loading data ===
(X_train, train_t), (X_test, test_t) = load_mnist(normalize=True)

# data reduction
X_train = X_train[:1000]
train_t = train_t[:1000]
train_t = to_categorical(train_t)

# setting parameters
max_epochs = 20
train_size = X_train.shape[0]
batch_size = 200
learning_rate = 0.01
scale = 0.005
iter_per_epoch = max(train_size / batch_size, 1)
max_iter = int(max_epochs * iter_per_epoch)

# constructing model
model = Neural_net(n_input=784, n_hidden=[50, 100, 70, 100],
                   n_output=10, w_std=scale, batch_norm=False)

bn_model = Neural_net(n_input=784, n_hidden=[50, 100, 70, 100],
                   n_output=10, w_std=scale, batch_norm=True)



model_dict = {'batch_norm':bn_model, 'normal':model}
train_acc = {}


'''---- learning ----'''
for name, network_ in model_dict.items():
    network_.add(['relu'] * 4)
    optim = Adam(lr=learning_rate)
    network_.compile(loss='categorical_crossentropy', optimizer=optim)
    hist = network_.fit(X_train, train_t,
                        n_iter=max_iter,
                        batch_size=batch_size,
                        history=True)
    train_acc[name] = np.asarray(hist['acc'])


# drawing graph
x = np.arange(max_epochs)
idx = np.arange(0, max_iter, iter_per_epoch)
idx = list(map(int,idx))

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)

ax.plot(x, train_acc['batch_norm'][idx],
        label='Batch Normalization', markevery=2, color='r')
ax.plot(x, train_acc['normal'][idx], linestyle = "--",
        label='Normal (without BatchNorm)', markevery=2, color='blue')

plt.legend(fontsize=15)
plt.title('Training accuracy', fontsize=20)
plt.xlabel('epochs',fontsize=20)
plt.xticks(np.arange(0,max_epochs+1,5),fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0,1.1)
plt.tight_layout()
plt.show()
