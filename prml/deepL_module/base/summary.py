from deepL_module.nn.layers import *
from deepL_module.nn.sequential import *
import joblib
import pandas as pd


def seq_save_model(model, path:str=None, name:str=None):

    feature = {'params':model.params, 'units':model.units_num}
    feature['layers'] = {'name':[],'args':[]}

    for layer in model.layers:
        feature['layers']['name'].append(layer.__class__.__name__)

        _args = None
        if isinstance(layer, (Conv2D,Maxpooling)):
            _args = layer.cache
        elif isinstance(layer, Dense):
            _args = [layer.units, layer.input_dim, layer.act_func]
        elif isinstance(layer, Activation):
            _args = layer.func
        else:
            _args = [None]

        feature['layers']['args'].append(_args)

    feature['conv_params'] = model.conv_params
    feature['idx'] = model.idx
    feature['batch_num'] = model.batch_num
    feature['cost_function'] = model.cost_function
    feature['alpha'] = model.alpha
    feature['wscale'] = model.wscale

    save_model(feature, path , name)

def seq_load_model(path:str=None):
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))
        path += '/../datasets/model_data/test.pkl'

    feat = loads(path)
    model = Sequential(alpha=feat['alpha'], w_std=feat['wscale'])
    model.params = feat['params']
    model.units_num = feat['units']

    for name, args in zip(feat['layers']['name'], feat['layers']['args']):

        if name == 'Activation':
            layer_instance = args
        else:
            layer_instance = eval(name + '(*args)')

        model.layers.append(layer_instance)

    w_loc = np.where([isinstance(obj,(Dense,Conv2D)) for obj in model.layers])[0]

    for n,idx in enumerate(list(w_loc), 1):
        args = [feat['params']['W' + str(n)], feat['params']['b' + str(n)]]
        model.layers[idx].set_param(*args)

    model.conv_params = feat['conv_params']
    model.idx = feat['idx']
    model.batch_num = feat['batch_num']
    model.cost_function = feat['cost_function']

    return model


def print_seq_summary(model, line_length=None, positions=None):

    Layers = model.layers.copy()

    line_length = line_length or 69
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


    def get_shapes(layers):
        shapes_list = []
        for n,layer in enumerate(layers):
            shape_val = None
            if isinstance(layer,Activation) and n > 0:
                shape_val = shapes_list[n-1]
            else:
                shape_val = list(layer.out.shape)
                shape_val[0] = None

            #list(shape_val) = None
            shapes_list.append(tuple(shape_val))
        return shapes_list

    def get_params(layers):
        params_list = []

        for layer in layers:
            num_param = 0
            if isinstance(layer, (Dense, Batch_norm_Layer, Conv2D)):
                num_param = layer.W.size + layer.b.size

            params_list.append(num_param)

        return params_list

    def get_cls(layers):

        name_trans = {
                      'Dense':{'name':'DenseLayer', 'value':0},
                      'Activation':{'name':'Activation', 'value':0},
                      'Conv2D':{'name':'Convol2dim', 'value':0},
                      'Maxpooling':{'name':'Maxpooling', 'value':0},
                      'Batch_norm_Layer':{'name':'Batch_Norm', 'value':0},
                      'Dropout_Layer':{'name':'-> Dropout', 'value':0}
                      }

        layers_list = []

        for layer in layers:

            cls_name = layer.__class__.__name__
            label = name_trans.get(cls_name)

            if label == None:
                label = name_trans['Activation']

            label['value'] += 1
            name = label['name'] + '_{}'.format(label['value'])
            layers_list.append('{}  ({})'.format(name, cls_name.split('_')[0]))

        return layers_list

    print('_' * line_length)
    to_display = ['Layer (type)', 'Output shape', 'Param #']
    print_row(to_display, positions)
    print('=' * line_length)

    df = np.array([get_cls(Layers), get_shapes(Layers), get_params(Layers)]).T


    for n in range(df.shape[0]):
        fields = df[n]
        print_row(fields, positions)
        if n < df.shape[0]-1:
            print('-' * line_length)

    print('=' * line_length)
    print( 'Total params: ' + str(np.sum(get_params(Layers))) )
    print('Optimizer: ' + str(model.optim.__class__.__name__))
