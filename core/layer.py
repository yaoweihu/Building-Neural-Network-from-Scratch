import numpy as np


class Layer:

    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Dense(Layer):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.params = {'w': np.random.randn(dim_out, dim_in),
                       'b': np.zeros(dim_out)}
        self.grads = {}
        self.cxt = {}

    def forward(self, input):
        self.cxt['x'] = input
        return input @ self.params['w'].T + self.params['b']

    def backward(self, grad):
        self.grads['w'] = grad.T @ self.cxt['x']
        self.grads['b'] = np.sum(grad, axis=0)
        return grad @ self.params['w']