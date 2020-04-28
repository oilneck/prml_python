import numpy as np
import pickle
import os

def get_mini_batch(train_x, train_t, batch_size=None):

    if batch_size is None:
        return train_x, train_t
    else:
        train_size = train_x.shape[0]
        batch_mask = np.random.choice(train_size, batch_size)
        return train_x[batch_mask], train_t[batch_mask]


def to_categorical(t,cls_num:int=None):
    if cls_num is None:
        cls_num = np.max(t) + 1
    return np.identity(cls_num)[t]


def smooth_filt(x):
     window_len = 11
     w = np.ones(window_len)
     y = np.convolve(w / w.sum(), x, mode='valid')
     margin_r = x[-(window_len-1):]
     margin_mean = [np.mean(margin_r,axis=-1)] * len(margin_r)
     y = list(y) + margin_mean
     return np.asarray(y)


def save_model(model, path:str=None):
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))
        path += '/../datasets/model_data/test_model.pkl'

    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path:str=None):
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))
        path += '/../datasets/model_data/test_model.pkl'

    with open(path, 'rb') as f:
        obj = pickle.load(f)

    return obj


def print_summary(model, line_length=None, positions=None):

    line_length = line_length or 67
    positions = positions or [.5, .85, 1.]

    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]

    def print_layer(layer):
        to_display = ['Input Layer', model.n_input, 0]
        print_row(to_display, positions)
        print('_' * line_length)

        params_ = np.asarray(count_params(model.params))
        for n, (key, val) in enumerate(layer.items()):

            name = key
            type_ = val.__class__.__name__.split('_')[0]

            if 'layer_' in key:
                name = 'DenseLayer_' + key[-1]


            shape_ = list(np.repeat(np.asarray(model.n_hidden_list), 2))
            shape_ += [model.n_output]

            n_param = params_[int(key[-1])-1]

            if 'activation_' in key:
                n_param = 0

            to_display = [name + '  ({})'.format(type_), shape_[n], n_param]
            print_row(to_display, positions)
            print('_' * line_length)

        # print activation of output layers
        name = '{0}'.format(model.cost_function.__class__.__name__)
        if name == 'Sum_squared_error':
            name = 'Linear'
        fields = 'Output Layer' + '  (' + name.split('_')[0] + ')'
        to_display = [fields, model.n_output, 0]
        print_row(to_display, positions)
        print('=' * line_length)
        print('Total params: ' + str(params_.sum()))
        print('Optimizer: ' + str(model.optim.__class__.__name__))


    def print_row(fields, positions):
        line = ''
        for i in range(len(fields)):
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)

    def count_params(w_dict):

        node_num = model.total_hidden_num + 2
        params_list = []

        for idx in range(1, node_num):
            n_weight = w_dict['W'+str(idx)].size
            n_bias = w_dict['b'+str(idx)].size
            params_list.append(n_weight + n_bias)

        return params_list

    print('_' * line_length)
    to_display = ['Layer (type)', 'Num Unit', 'Param #']
    print_row(to_display, positions)
    print('=' * line_length)

    print_layer(model.layers)
