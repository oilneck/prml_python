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


def save_model(model, path:str=None, name:str=None):
    if path is None and name is None:
        path = os.path.dirname(os.path.abspath(__file__))
        path += '/../datasets/model_data/test_model.pkl'
    elif path is None and name is not None:
        path = os.path.dirname(os.path.abspath(__file__))
        path += '/../datasets/model_data/' + name + '.pkl'
    elif path is not None and name is None:
        pass
    else:
        raise Exception("Could not specify 'path' and 'name'")

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


    def print_row(fields, positions):
        line = ''
        for i in range(len(fields)):
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)


    def count_params(layer_dict):
        params_list = []

        for key, val in layer_dict.items():
            num_param = 0 # the number of params in activation layer has zero
            if 'DenseLayer_' in key or 'Batch_Norm_' in key:
                num_param = val.n_param
            params_list.append(num_param)

        return params_list


    def count_shapes(layer_dict):
        shapes_list = []

        for layer in layer_dict.values():
            shape_val = list(layer.out.shape)
            shape_val[0] = None
            shapes_list.append(tuple(shape_val))

        return shapes_list

    params_ = count_params(model.layers)
    shapes_ = list(count_shapes(model.layers)) + [model.n_output]

    def print_layer(layer):

        to_display = ['Input Layer', model.n_input, 0]
        print_row(to_display, positions)
        stack_id = 0

        for n, (name, val) in enumerate(layer.items()):

            identifier = int(name[-1])
            if identifier > stack_id:
                stack_id = identifier
                print('_' * line_length)

            type_ = val.__class__.__name__.split('_')[0]
            to_display = [name + '  ({})'.format(type_), shapes_[n], params_[n]]

            print_row(to_display, positions)


        # print activation of output layers
        name = model.cost_function.__class__.__name__
        if name == 'Sum_squared_error':name = 'Linear'
        fields = 'Output Layer' + '  (' + name.split('_')[0] + ')'

        to_display = [fields, model.n_output, 0]
        print_row(to_display, positions)



    print('_' * line_length)
    to_display = ['Layer (type)', 'Output shape', 'Param #']
    print_row(to_display, positions)
    print('=' * line_length)

    print_layer(model.layers)

    print('=' * line_length)
    print( 'Total params: ' + str(np.sum(params_)) )
    print('Optimizer: ' + str(model.optim.__class__.__name__))

def calc_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
