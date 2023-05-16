from __future__ import annotations
import numpy as np 
from typing import NamedTuple, Callable, Optional, Union

class Dependency(NamedTuple):
	tensor: Tensor
	grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, np.ndarray]
def ensure_array(arrayable: Arrayable) -> np.ndarray:
	return arrayable if isinstance(arrayable, np.ndarray) else np.array(arrayable)

class Tensor:
	def __init__(self, data: Arrayable, requires_grad: bool = False, depends_on = None) -> None:
		self.data = ensure_array(data)
		self.requires_grad = requires_grad
		self.depends_on = depends_on or []
		self.shape = self.data.shape
		self.grad: Optional[Tensor] = None

		if self.requires_grad:
			self.zero_grad()

	def __repr__(self) -> str:
		return f"Tensor({self.data}, requires_grad={self.requires_grad})" 

	def zero_grad(self) -> None:
		self.grad = Tensor(np.zeros_like(self.data))

	def backward(self, grad: Tensor = None) -> None:
		assert self.requires_grad, "backward() called on non-grad tensor"

		if grad is None:
			if self.shape == ():
				grad = Tensor(1)
			else:
				raise RuntimeError("grad must be specified for non-0-tensor")

		self.grad.data += grad.data

		for dependency in self.depends_on:
			backward_grad = dependency.grad_fn(grad.data)
			dependency.tensor.backward(Tensor(backward_grad))

	def sum(self) -> Tensor:
		return tensor_sum(self)


# --- reduce ops --- 
def tensor_sum(t: Tensor) -> Tensor:
	data = t.data.sum()
	requires_grad = t.requires_grad
	depends_on = []
	if requires_grad:
		def grad_fn(grad: np.ndarray) -> np.ndarray:
			return grad * np.ones_like(t.data) # do we need np.ones_like?
		depends_on = [Dependency(t, grad_fn)]
	
	return Tensor(data, requires_grad, depends_on)
