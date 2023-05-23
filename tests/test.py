import numpy as np
from tensor import Tensor
from graph import draw

x = Tensor(np.eye(3), requires_grad=True)
y = Tensor(np.array([2.0, 0, -1.0]).reshape(1,-1), requires_grad=True)
z = y.matmul(x)
loss = z.sum()

# draw(loss)
loss.backward()
print(x.grad.data)
print(y.grad.data)


import torch
x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0, 0, -1.0]], requires_grad=True)
z = y.matmul(x)
loss = z.sum()
loss.backward()
print(x.grad.data)
print(y.grad.data)
