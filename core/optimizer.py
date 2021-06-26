import numpy as np


class Optimizer:

    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, grads, params):
        for grad, param in zip(grads, reversed(params)):
            for key in param.keys():
                
                new_grad = self.compute_step(grad[key])
                if self.weight_decay:
                    new_grad += self.weight_decay * param[key]
                param[key] -= self.lr * new_grad
        
    def compute_step(self, grads):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, lr=0.01, weight_decay=0.0):
        super().__init__(lr, weight_decay)

    def compute_step(self, grads):
        return grads 