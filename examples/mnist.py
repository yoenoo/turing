import numpy as np 
from turing.tensor import Tensor
import turing.nn as nn
from turing.optim import SGD
from turing.utils.data.mnist import fetch_mnist
from tqdm import trange

bs = 128
epochs = 1000

# load the MNIST data
X_train, Y_train, X_test, Y_test = fetch_mnist()
#X_train = Tensor(X_train / 255.)
#Y_train = Tensor(Y_train, dtype=np.int8)
#X_test = Tensor(X_test / 255.)
#Y_test = Tensor(Y_test, dtype=np.int8)

class TinyNet(nn.Module):
  def __init__(self):
    self.fc1 = nn.Linear(28*28, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    out = self.fc1(x)
    out = out.relu()
    out = self.fc2(out)
    return out

model = TinyNet()

lr = 0.01
num_classes = int(Y_train.max()+1)
assert num_classes == 10

optimizer = SGD(model.parameters(), lr=lr)
cross_entropy_loss = nn.CrossEntropyLoss()

losses, accuracies = [], [] 
for epoch in (t := trange(epochs)):
  samp = np.random.randint(0, X_train.shape[0], size=(bs,))
  x = Tensor(X_train[samp]).reshape(-1,28*28) # TODO: Tensor(X_train[samp]) vs Tensor(X_train)[samp] -> Tensor.__getitem__
  Y = Y_train[samp]

  optimizer.zero_grad()
  logits = model.forward(x)
  loss = cross_entropy_loss(logits, Y, num_classes=num_classes)
  loss.backward()
  optimizer.step()

  cat = np.argmax(logits.data, axis=1)
  acc = (cat == Y.data).mean()
  t.set_description(f"loss {loss.data:.2f} accuracy {acc:.2f}")

y_test_preds_out = model.forward(Tensor(X_test).reshape(-1,28*28))
y_test_preds = np.argmax(y_test_preds_out.data, axis=1)
acc = (Y_test == y_test_preds).mean()
print(f"test set accuracy {acc:.4f}")
