import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from deepL_module.datasets.mnist import load_mnist
from deepL_module.nn.sequential import Sequential
from deepL_module.nn.optimizers import *
from deepL_module.base import *
from deepL_module.nn.layers import *


def show_filters(filters, fig_num, title:str, n_row=6):

    filter_num = filters.shape[0]
    n_col = int(np.ceil(filter_num / n_row))
    fig = plt.figure(fig_num,figsize=(8,5))
    patch = patches.Rectangle(xy=(0, 0), width=0.25, height=0.5, ec='#000000', fill=False)

    for i in range(filter_num):
        ax = fig.add_subplot(n_col, n_row, i+1)
        ax.imshow(filters[i,0], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.tick_params(bottom=False, left=False)
    fig.suptitle(title,fontsize=20)
    ax.add_patch(patch)
    plt.show()


'''#0 loading data '''
(X_train, train_t), (X_test, test_t) = load_mnist(flatten=False)
train_t = to_categorical(train_t)


'''#1 config for NN '''
model = Sequential(w_std=0.01)
model.add(Conv2D(30,(5,5),input_shape=(1,28,28)))
model.add(Activation('relu'))
model.add(Maxpooling(pool_h=2, pool_w=2, stride=2))
model.add(Dense(100, activation='relu'))
model.add(Dense(10))

# optimizer setting
routine = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=routine)

'''#2 visualizing initial params '''
show_filters(model.params['W1'], fig_num=1, title='before learning')

'''#3 learning '''
# hist = model.fit(X_train, train_t,
#                  batch_size=256,
#                  epochs=30,
#                  history=True)

#model.save(name='visualize_filter_CNN')

path_r = './../../../../prml/deepL_module/datasets/model_data/visualize_filter_CNN.pkl'
model = load_model(path_r)

'''#4 visualizing params after learning'''
show_filters(model.params['W1'], fig_num=2, title='after learning')
