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
