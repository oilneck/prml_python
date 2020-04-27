import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from deepL_module.nn.multi_layer_nn import Neural_net
from deepL_module.nn.optimizers import *
from deepL_module.nn.cost_functions import *
from deepL_module.base import *


'''#0 loading data '''
(X_train, train_t), (X_test, test_t) = load_mnist(normalize=True)

train_size = X_train.shape[0]
batch_size = 256
max_iter = 1e+4


'''#1 config for NN '''
model = Neural_net(784, [100, 100, 100, 100], 10, alpha=0.01)
model.add(['tanh', 'softsign', 'softplus', 'swish'])
model.set_loss('categorical_crossentropy')
train_loss = []
train_acc = []


'''#2 optimizer '''
optim = Adam(lr=0.01)


'''#3 learning '''
for i in range(int(max_iter)):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = X_train[batch_mask]
    t_batch = to_categorical(train_t[batch_mask], cls_num=10)

    # update weight matrix
    grads = model.gradient(x_batch, t_batch)
    optim.update(model.params, grads)

    # get loss data
    loss = model.loss(x_batch, t_batch)
    train_loss.append(loss)

    # get accuracy data
    score = model.accuracy(x_batch, t_batch)
    train_acc.append(score)


'''#4 drawing loss & accuracy data '''
# ----- loss data -----
fig1 = plt.figure(figsize=(13,5)) # dpi=50
ax = fig1.add_subplot(121)
x = np.arange(max_iter)
ax.plot(x, train_loss, color='blue',marker="*", markersize=7, markevery=10)
plt.xlabel("iterations",fontsize=25)
plt.title("loss",fontsize=25)
plt.xlim([-5,210])
plt.show()

# --- accuracy data ---
ax = fig1.add_subplot(122)
ax.plot(x, smooth_filt(train_acc),
        color = 'lime',
        marker = "*",
        markersize = 8,
        markevery = 100)
plt.xlabel("iterations",fontsize=25)
plt.title("accuracy",fontsize=25)
plt.xlim([-30,1000])
plt.tight_layout()
plt.show()



'''#5 preparing test data '''
test_size = X_test.shape[0]
fetch_idx = np.random.choice(test_size, 1)
data = X_test[fetch_idx]
label = test_t[fetch_idx]


'''#6 showing image '''
fig2 = plt.figure(figsize=(11,4))
ax = fig2.add_subplot(111)
ax.imshow(data.reshape(28,28),cmap='gray')
plt.tick_params(labelbottom=False,
                labelleft=False,
                bottom=False,
                left=False)


'''#7 output prediction data '''
prob = model(data).ravel()
prediction = np.argmax(prob)

# --- probability ---
c_list = ['k'] * 10
c_list[prediction] = 'r'
for n in range(len(prob)):
    p = np.round(prob[n],3)
    text = '{}:  {:.2g}'.format(n,p)
    fig2.text(0.1, 0.93-0.1*n, text, color=c_list[n], size=15)

# --- prediction ---
pos = ax.get_position()
pos_y = 0.5 * (pos.y1 - pos.y0)
fig2.text(0.75, pos_y, str(prediction), fontsize=60, color='r')
fig2.text(0.71,0.65, "prediction",
        fontsize = 20,
        transform = fig2.transFigure,
        color = 'r')

# --- labels ---
fig2.text(0.87, pos_y, str(label[0]), fontsize=60, color='k')
fig2.text(0.86,0.65, "labels",
        fontsize = 20,
        transform = fig2.transFigure)
plt.show()


'''#8 save model'''
path = './../../prml/deepL_module/datasets/model_data/im_model.pkl'
save_model(model, path)
