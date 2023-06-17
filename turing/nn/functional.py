import numpy as np 
from turing.tensor import Tensor

def one_hot(tensor, num_classes=-1):
	if tensor.ndim != 1:
		raise RuntimeError(f"does not support {tensor.ndim}-dim tensor")
	n = len(tensor)
	out = np.zeros((n, num_classes))
	out[np.arange(n), tensor.data] = 1
	return Tensor(out, requires_grad=False)
