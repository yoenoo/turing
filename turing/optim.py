class Optimizer:
  def __init__(self, params, lr):
    self.params = list(params)
    self.lr = lr

class SGD(Optimizer):
  def __init__(self, params, lr):
    super().__init__(params, lr)

  def zero_grad(self):
    for p in self.params:
      if p.grad is not None:
        p.grad = None

  def step(self):
    for p in self.params:
      p.data -= p.grad.data * self.lr
