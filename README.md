# Turing Project
A lightweight tensor / autograd library entirely built on top of NumPy, with PyTorch-like syntax.

The Tensor class is a wrapper around a NumPy array, expect it does Tensor things.

### Example
```python
from turing.tensor import Tensor
import tinygrad.nn as nn
from turing.optim import SGD

class TinyNet(nn.Module):
  def __init__(self):
    self.fc1 = nn.Linear(28*28, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    # can also do x.matmul(w).relu().matmul(w2)
    out = self.fc1(x)
    out = out.relu()
    out = self.fc2(out)
    return out

model = TinyNet()
optimizer = optim.SGD(model.parameters(), lr=0.001)
cross_entropy_loss = nn.CrossEntropyLoss()

# ... complete data loader
optimizer.zero_grad()
logits = model.forward(x) 
loss = cross_entropy_loss(logits, y_true, num_classes)
loss.backward()
optimizer.step()
```

### TODO
- [ ] Add tests for more ops
- [ ] Add CIFAR-10 example
- [ ] Add DAG computation graph 
- [ ] Implement nn.Sequential
- [ ] Implement other optimizers e.g. Momentum, RMSProp, Adam
- [ ] Implement CNN using [im2col](https://github.com/3outeille/CNNumpy/blob/5394f13e7ed67a808a3e39fd381f168825d65ff5/src/fast/utils.py#L360)
- [ ] Implement RNN/LSTM

## Installation
The current recommended way to install turing is from source.

### From source
```sh
git clone https://github.com/yoenoo/turing.git
cd turing
pip install -e .
```

### From PyPI
check again later...
