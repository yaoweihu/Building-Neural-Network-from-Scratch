import copy


class Net:

    def __init__(self, layers):
        self.layers = layers

    def __repr__(self):
        return '\n'.join([str(layer) for layer in self.layers])

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, grad):
        layer_grads = []
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            layer_grads.append(copy.copy(layer.grads))
        return layer_grads

    @property
    def params(self):
        return [layer.params for layer in self.layers]


class Model:

    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, input):
        return self.net.forward(input)

    def backward(self, pred, target):
        loss = self.loss.loss(pred, target)
        grad = self.loss.grad(pred, target)
        grad = self.net.backward(grad)
        return loss, grad

    def apply_grads(self, grad):
        params = self.net.params
        self.optimizer.step(grad, params)

