from deepL_module.nn.layers import *
from deepL_module.nn.sequential import *
import joblib


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
