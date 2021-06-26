import numpy as np


class Loss:

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def grad(self, *args, **kwargs):
        raise NotImplementedError


class MSE(Loss):

    def loss(self, preds, targets):
        return 0.5 * np.mean((preds - targets) ** 2)

    def grad(self, preds, targets):
        return (preds - targets) / preds.shape[0]


class MAE(Loss):

    def loss(self, preds, targets):
        return np.mean(np.abs(preds - targets))

    def grad(self, preds, targets):
        return np.sign(preds - targets) / preds.shape[0]