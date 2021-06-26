import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt

from core import *


x = np.linspace(-1, 1, 100).reshape(-1, 1)
y = x ** 2 + 0.2 * np.random.randn()


net = Net([Dense(1, 30), ReLU(), Dense(30, 1)])
loss = MSE()
sgd = SGD(lr=0.1)
model = Model(net, loss, sgd)

for t in range(1000):

    pred = model.forward(x)
    loss, grad = model.backward(pred, y)
    model.apply_grads(grad)

    print(f'Epoch: {t}, Loss: {loss:.5f}')


plt.figure(figsize=(10,4))
plt.scatter(x, y, color = "orange")
plt.plot(x, model.forward(x), color='blue', linewidth=3)
plt.title('Regression Analysis')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()