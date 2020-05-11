from deepL_module.base import *
import matplotlib.pyplot as plt


def show_filters(filters, n_row=6):

    filter_num = filters.shape[0]
    n_col = int(np.ceil(filter_num / n_row))
    fig = plt.figure(figsize=(8,4))

    for i in range(filter_num):
        ax = fig.add_subplot(n_col, n_row, i+1)
        ax.imshow(filters[i,0], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.tick_params(bottom=False, left=False)
    plt.tight_layout()
    plt.show()

path_r = './../../../../prml/deepL_module/datasets/model_data/simple_CNN.pkl'
model = load_model(path_r)
show_filters(model.params['W1'])
