import numpy as np
from tensor import Tensor
from utils import fetch_mnist

import torch
class TinyNet:
	def __init__(self):
		self.l1 = torch.randn(784, 128)
		self.l2 = torch.randn(128, 10)

	def forward(self, x):
		x = x.dot(self.l1)
		x = x.relu()
		x = x.dot(self.l2)
		x = x.logsoftmax()
		return x


X_train, Y_train, X_test, Y_test = fetch_mnist()

net = TinyNet()
net.forward(torch.tensor(X_train[range(5)].reshape(5, -1)))
