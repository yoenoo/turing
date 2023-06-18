import torch
import numpy as np 
from turing.tensor import Tensor
import turing.nn.functional as F
from turing.utils.data.mnist import fetch_mnist

bs = 128
epochs = 1000

# load the MNIST data
X_train, Y_train, X_test, Y_test = fetch_mnist()
#X_train = Tensor(X_train / 255.)
#Y_train = Tensor(Y_train, dtype=np.int8)
#X_test = Tensor(X_test / 255.)
#Y_test = Tensor(Y_test, dtype=np.int8)

"""
  x = Tensor.randn(32,28*28)
  w1 = Tensor.randn(28*28,128).requires_grad_()
  w2 = Tensor.randn(128,10).requires_grad_()
  print(x)
  print(w1) print(w2)

  out = x.matmul(w1).relu().matmul(w2).logsoftmax()
  print(out.data)
  out.backward()

  print(w1.grad)
  print(w2.grad)

  #out = x.matmul(y)
"""
def linear(nin, nout):
  #w = (Tensor.rand(nin, nout) * 2 - 1) * nin ** -0.5 / np.sqrt(nin*nout)
  w = (Tensor.rand(nin, nout) * 2 - 1) / np.sqrt(nin*nout)
  w.requires_grad_()
  b = (Tensor.rand(1, nout) * 2 - 1) * nin ** -0.5
  b.requires_grad_()
  return w, b

"""
w1 = (Tensor.rand(28*28,128) * 2 - 1) / 28
w1.requires_grad_()
w2 = (Tensor.rand(128,10) * 2 - 1) / np.sqrt(128)
w2.requires_grad_()
"""
w1, b1 = linear(28*28, 128)
w2, b2 = linear(128, 10)

def layer_init(m, h):
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)

class BobNet:
  def __init__(self):
    self.l1 = Tensor(layer_init(784,128), requires_grad=True)
    self.l2 = Tensor(layer_init(128,10), requires_grad=True)

  def forward(self, x):
    return x.matmul(self.l1).relu().matmul(self.l2).logsoftmax()

model = BobNet()

lr = 0.01
num_classes = int(Y_train.max()+1)
losses, accuracies = [], [] 
for epoch in range(epochs):
  samp = np.random.randint(0, X_train.shape[0], size=(bs,))
  x = Tensor(X_train[samp].reshape(-1, 28*28))
  Y = Y_train[samp]
  y = F.one_hot(Y, num_classes) 

  # zero grad
  """
  if w1.grad is not None:
    w1.grad = None #Tensor.zeros(*w1.shape)
    #w1.grad.data = np.zeros(w1.shape)
  if w2.grad is not None:
    w2.grad = None #Tensor.zeros(*w2.shape)
    #w2.grad.data = np.zeros(w2.shape)
  """
  model.l1.grad = None
  model.l2.grad = None

  # out = x.matmul(w1).relu().matmul(w2).logsoftmax()
  out = model.forward(x)
  loss = (-out).mul(y).mean() 
  loss.backward()

  """
  x = torch.tensor(x.data, requires_grad=False)
  w1 = torch.tensor(w1.data, requires_grad=True)
  w2 = torch.tensor(w2.data, requires_grad=True)
  out = x.matmul(w1).relu().matmul(w2)
  from torch.nn import CrossEntropyLoss
  cel = CrossEntropyLoss()
  loss = cel(out, torch.tensor(Y.data, dtype=torch.long))
  """

  #print(w1.grad)
  #print(w2.grad)
  
  # SGD
  #w1.data -= w1.grad.data * lr
  #w2.data -= w2.grad.data * lr
  #print(w1.grad.min(), w1.grad.max())
  #print(w2.grad.min(), w2.grad.max())
  model.l1.data = model.l1.data - lr * model.l1.grad.data
  model.l2.data = model.l2.data - lr * model.l2.grad.data

  cat = np.argmax(out.data, axis=1)
  acc = (cat == Y.data).mean()
  #print(epoch, loss.item(), acc)
  print(epoch, loss.data, acc)

#y_test_preds_out = Tensor(X_test.reshape(-1,28*28)).matmul(w1).relu().matmul(w2).logsoftmax()
y_test_preds_out = model.forward(Tensor(X_test.reshape(-1,28*28)))
y_test_preds = np.argmax(y_test_preds_out.data, axis=1)
acc = (Y_test == y_test_preds).mean()
print(f"test set accuracy {acc:.4f}")
