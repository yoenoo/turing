import unittest
import random
import numpy as np 

import torch
from turing.tensor import Tensor

def set_seed(seed, deterministic=False):
  torch.use_deterministic_algorithms(deterministic)
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

def compare(s, x, y, atol, rtol):
  assert isinstance(x, torch.Tensor)
  assert isinstance(y, Tensor)
  assert x.shape == y.shape, f"shape mismatch: turing={x.shape} | torch={y.shape}"
  try:
    np.testing.assert_allclose(x.detach().numpy(), y.data, atol=atol, rtol=rtol)
  except Exception:
    raise Exception(f"{s} failed shape {x.shape}")

def helper_test_op(vals, torch_fn, turing_fn, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3, forward_only=False):
  set_seed(42)

  ts = [torch.tensor(v, requires_grad=True) for v in vals]
  tst = [Tensor(v, requires_grad=True) for v in vals]

  ts_out = torch_fn(*ts)
  tst_out = turing_fn(*tst)

  compare("forward pass", ts_out, tst_out, atol=atol, rtol=rtol)

  if not forward_only: 
    (ts_out+1).square().mean().backward()
    (tst_out+1).square().mean().backward()

    for i, (t, tt) in enumerate(zip(ts, tst)):
      compare(f"backward pass tensor {i}", t, tt, atol=grad_atol, rtol=grad_rtol)

class TestTensorOps(unittest.TestCase):
  def test_add(self):
    vals = [np.random.randint(0,100,3).astype(np.float32), np.random.randint(0,100,3).astype(np.float32)]
    helper_test_op(vals, lambda x,y: x+y, Tensor.add)

  def test_matmul(self):
    vals = [np.random.randn(4,32), np.random.randn(32,10)] 
    helper_test_op(vals, torch.matmul, Tensor.matmul)

  def test_relu(self):
    vals = [np.random.randn(32,)]
    helper_test_op(vals, torch.relu, Tensor.relu)

  def test_sum(self):
    vals = [np.random.randn(4,32)] 
    helper_test_op(vals, torch.sum, Tensor.sum, forward_only=True)
