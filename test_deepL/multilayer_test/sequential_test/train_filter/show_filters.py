from deepL_module.base import *
import matplotlib.pyplot as plt

path_r = './../../../../prml/deepL_module/datasets/model_data/visualize_filter_CNN.pkl'
model = load_model(path_r)

filters = model.params['W1']
filter_num = filters.shape[0]
n_x = 6
n_y = int(np.ceil(filter_num / n_x))

fig = plt.figure()

for i in range(filter_num):
    ax = fig.add_subplot(n_y, n_x, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i,0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.tick_params(bottom=False, left=False)
plt.tight_layout()
plt.show()
