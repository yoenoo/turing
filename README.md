![Unit Tests](https://github.com/yoenoo/turing/actions/workflows/turing_test.yaml/badge.svg)

# Turing Project
A lightweight tensor / autograd library entirely built on top of NumPy, with PyTorch-like syntax.

The Tensor class is a wrapper around a NumPy array, expect it does Tensor things.

### Example
```python
import turing.nn as nn
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
optimizer = SGD(model.parameters(), lr=0.001)
cross_entropy_loss = nn.CrossEntropyLoss()

# ... complete data loader
optimizer.zero_grad()
logits = model.forward(x) 
loss = cross_entropy_loss(logits, y_true, num_classes)
loss.backward()
optimizer.step()
```

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
