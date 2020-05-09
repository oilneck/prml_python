import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from deepL_module.nn.sequential import Sequential
from deepL_module.nn.multi_layer_nn import *
from deepL_module.nn.optimizers import *
from deepL_module.base import *
from deepL_module.nn.layers import *
import copy

# === loading data ===
(X_train, train_t), (X_test, test_t) = load_mnist(normalize=True)

# data reduction
X_train = X_train[:1000]
train_t = train_t[:1000]
train_t = to_categorical(train_t)

# setting parameters
max_epochs = 20
batch_size = 200
learning_rate = 0.01
scale = 0.005

# constructing model
model = Sequential(w_std=scale)
model.add(Dense(50, input_dim=784, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10))


bn_model = Sequential(w_std=scale)

bn_model.add(Dense(50, input_dim=784))
bn_model.add(Batch_norm_Layer())
bn_model.add(Activation('relu'))

bn_model.add(Dense(100))
bn_model.add(Batch_norm_Layer())
bn_model.add(Activation('relu'))

bn_model.add(Dense(70))
bn_model.add(Batch_norm_Layer())
bn_model.add(Activation('relu'))

bn_model.add(Dense(100))
bn_model.add(Batch_norm_Layer())
bn_model.add(Activation('relu'))

bn_model.add(Dense(10))


model_dict = {'batch_norm':bn_model, 'normal':model}
train_acc = {}


'''---- learning ----'''
for name, network_ in model_dict.items():
    optim = Adam(lr=learning_rate)
    network_.compile(loss='categorical_crossentropy', optimizer=optim)
    hist = network_.fit(X_train, train_t,
                        epochs=max_epochs,
                        batch_size=batch_size,
                        history=True)
    train_acc[name] = np.asarray(hist['acc'])


# drawing graph
x = np.arange(max_epochs)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)

ax.plot(x, train_acc['batch_norm'],
        label='Batch Normalization', markevery=2, color='r')
ax.plot(x, train_acc['normal'], linestyle = "--",
        label='Normal (without BatchNorm)', markevery=2, color='blue')

plt.legend(fontsize=15)
plt.title('Training accuracy', fontsize=20)
plt.xlabel('epochs',fontsize=20)
plt.xticks(np.arange(0,max_epochs+1,5),fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0,1.1)
plt.tight_layout()
plt.grid()
plt.show()
