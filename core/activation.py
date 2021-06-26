import numpy as np
from .layer import Layer


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class Activation(Layer):

    def __init__(self):
        super().__init__()
        self.cxt = {}

    def forward(self, input):
        self.cxt['x'] = input
        return self.func(input)

    def backward(self, grad):
        return self.derivative(self.cxt['x']) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(Activation):

    def func(self, x):
        return sigmoid(x)

    def derivative(self, x):
        return sigmoid(x) * (1. - sigmoid(x))


class Tanh(Activation):

    def func(self, x):
        return 2 * sigmoid(2 * x) - 1

    def derivative(self, x):
        return 1 - sigmoid(x) ** 2


class ReLU(Activation):

    def func(self, x):
        return np.maximum(0., x)

    def derivative(self, x):
        return x > 0