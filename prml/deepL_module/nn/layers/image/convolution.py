import numpy as np
from deepL_module.base.util import *

class Conv2D(object):

    def __init__(self,filters:int, kernel_size:tuple,stride=1, pad=0, input_shape:tuple=(1,28,28)):
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.stride = stride
        self.pad = pad

        self.cache = [self.filters, self.kernel_size, input_shape, stride, pad]

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def set_param(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        self.out = out

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
