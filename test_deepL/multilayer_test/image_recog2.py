import matplotlib.pyplot as plt
from deepL_module.datasets.mnist import load_mnist
from deepL_module.base import *


'''#0 load training data '''
(X_train, train_t), (X_test, test_t) = load_mnist(normalize=True)

'''#1 load model data '''
path_r = './../../prml/deepL_module/datasets/model_data/im_model.pkl'
model = load_model(path_r)


'''#2 preparing test data '''
test_size = X_test.shape[0]
fetch_idx = np.random.choice(test_size, 1)
data = X_test[fetch_idx]
label = test_t[fetch_idx]


'''#3 showing image '''
fig = plt.figure(figsize=(11,4))
ax = fig.add_subplot(111)
ax.imshow(data.reshape(28,28), cmap='gray')
plt.tick_params(labelbottom = False,
                labelleft = False,
                bottom = False,
                left = False)


'''#4 output prediction data '''
prob = model.predict(data)
prediction = np.argmax(prob)

# --- probability ---
c_list = ['k'] * 10
c_list[prediction] = 'r'
for n in range(len(prob)):
    p = np.round(prob[:,n], 3)
    text = '{}:  {:.2g}'.format(n,float(p))
    fig.text(0.1, 0.93-0.1*n, text, color=c_list[n], size=15)

# --- prediction ---
pos = ax.get_position()
pos_y = 0.5 * (pos.y1 - pos.y0)
fig.text(0.75, pos_y, str(prediction), fontsize=60, color='r')
fig.text(0.71,0.65, "prediction",
        fontsize=20,
        transform=fig.transFigure,
        color='r')

# --- labels ---
fig.text(0.87, pos_y, str(label[0]), fontsize=60, color='k')
fig.text(0.86,0.65, "labels",
        fontsize=20,
        transform=fig.transFigure)
plt.show()
