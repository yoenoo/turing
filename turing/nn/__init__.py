from turing.tensor import Tensor
import turing.nn.functional as F

# TODO: change __call__ to forward
# TODO: Parameter class (subclass of Tensor)

class Module:
  _parameters = []

  def zero_grad_(self):
    for p in self._parameters: 
      if p.grad is not None:
        p.grad = None

  def parameters(self):
    yield from self._parameters

  def add_param(self, x):
    self._parameters.append(x)

class Linear(Module):
  """
  Applies a linear transformation to the incoming data: y = xA.T + b
  """
  def __init__(self, in_features, out_features, bias=True, dtype=None):
    k = in_features**-1
    lower, upper = -k**0.5, k**0.5
    self.weight = Tensor.rand(low=lower, high=upper, size=(in_features, out_features)) 
    self.weight.requires_grad_()
    super().add_param(self.weight)
    if bias: 
      self.bias = Tensor.rand(low=lower, high=upper, size=(1, out_features))
      self.bias.requires_grad_()
      super().add_param(self.bias)
    else:
      self.bias = None

  def __call__(self, input):
    return F.linear(input, self.weight, self.bias)

class CrossEntropyLoss:
  def __init__(self):
    pass

  def __call__(self, logits, target, num_classes):
    target_one_hot = F.one_hot(target, num_classes=num_classes)
    log_softmax_logits = logits.logsoftmax()
    return (-log_softmax_logits).mul(target_one_hot).mean()


if __name__ == "__main__":
  class Net(Module):
    def __init__(self):
      self.fc1 = Linear(28*28, 128)
      self.fc2 = Linear(128, 10)

    def forward(self, x):
      return self.fc2(self.fc1(x).relu())

  net = Net()
  print(list(net.parameters()))
