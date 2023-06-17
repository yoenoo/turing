import warnings
import numpy as np 
from functools import partialmethod

DEBUG = False

#from enum import Enum, auto
#class DataType(Enum): f16 = auto(); f32 = auto(); f64 = auto()

class Tensor:
	def __init__(self, data, requires_grad=False, dtype=np.float32):
		if dtype == np.float64:
			warnings.warn("creating Tensor with dtype float64. this may result in slower computation")
		if isinstance(data, list):
			data = np.array(data, dtype=dtype)
		if isinstance(data, int):
			data = np.array(data)

		assert type(data) == np.ndarray, f"invalid data type: {type(data)}"

		self.data = data.astype(dtype)
		self.grad = None
		self.requires_grad = requires_grad
		self._ctx = None # used for autograd graph construction
		self.dtype = dtype

	def __repr__(self):
		#tensor(2.3029, grad_fn=<NegBackward0>)
		return f"<Tensor{self.shape} [requires_grad]>" if self.requires_grad else f"<Tensor{self.shape}>"

	def __iter__(self):
		yield self

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		out = self
		out.data = out.data[i]
		return out

	def __add__(self, x):
		assert isinstance(x, int) or isinstance(x, float)
		return self.add(Tensor(x))

	def __sub__(self, x):
		assert isinstance(x, int) or isinstance(x, float)
		return self.add(Tensor(-x))

	def __neg__(self):
		return self.mul(Tensor([-1]))
		#out = self
		#out.data *= -1
		#return out

	def __mul__(self, x):
		assert isinstance(x, int) or isinstance(x, float)
		return self.mul(Tensor([x]))

	def __truediv__(self, x):
		assert isinstance(x, int) or isinstance(x, float)
		return self.mul(Tensor([1/x]))	

	def min(self):
		return self.data.min()

	def max(self):
		return self.data.max()

	def requires_grad_(self):
		self.requires_grad = True
		return self

	@property
	def shape(self):
		return self.data.shape

	@property
	def ndim(self):
		return len(self.shape)

	def reshape(self, *shape):
		out = self	
		out.data = out.data.reshape(*shape)
		return out

	@staticmethod
	def zeros(*shape, dtype=np.float32):
		return Tensor(np.zeros(shape, dtype=dtype))

	@staticmethod
	def ones(*shape, dtype=np.float32):
		return Tensor(np.ones(shape, dtype=dtype))

	@staticmethod
	def rand(*shape, dtype=np.float32):
		return Tensor(np.random.uniform(0,1,shape).astype(dtype))

	@staticmethod
	def randn(*shape, dtype=np.float32):
		return Tensor(np.random.randn(*shape).astype(dtype))

	def deepwalk(self):
		def _deepwalk(node, visited, nodes):
			visited.add(node)
			if node._ctx:
				for i in node._ctx.parents:
					if i not in visited: 
						_deepwalk(i, visited, nodes)
				nodes.append(node)
			return nodes
		return _deepwalk(self, set(), [])

	def backward(self):
		if self.ndim != 0:
			raise RuntimeError("grad can be implicitly created only for scalar outputs")
		self.grad = Tensor(1, requires_grad=False)

		for t0 in reversed(self.deepwalk()):
			if DEBUG: print("\nops:", t0._ctx)
			if not any(x.requires_grad for x in t0._ctx.parents):
				continue

			assert t0.grad is not None
			grads = t0._ctx.backward(t0.grad.data)
			if DEBUG: print("t0.parents:", t0._ctx.parents)
			# grads = [Tensor(g, requires_grad=False) for g in ([grads] if len(t0._ctx.parents) == 1 else grads)] # grads don't need grads
			if DEBUG: print("grads:", grads)
			if DEBUG: print([g.data for g in grads])
			if any(not isinstance(g, Tensor) for g in grads): raise RuntimeError("invalid tensor type")
			for t, g in zip(t0._ctx.parents, grads):
				#print("ops_:", t._ctx)
				#print(t.data, g.data)
				if g is not None and t.requires_grad:
					assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
					t.grad = g if t.grad is None else (t.grad + g)

	"""
	def __sub__(self, x):
		out = self
		out.data -= x
		return out 

	def __truediv__(self, x):
		out = self	
		out.data /= x
		return out

	def __mul__(self, x):
		# do we really want this?
		#return self.mul(Tensor([x]))
		out = self
		out.data *= x
		return out
	"""

class Function:
	def __init__(self, *tensors):
		self.parents = tensors
		self.requires_grad = any(t.requires_grad for t in self.parents)

	def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
	def backward(self, *args, **kwargs): raise NotImplementedError(f"backward not implemented for {type(self)}")

	def apply(self, f, *x, **kwargs):
		x = list(self) + list(x) 
		ctx = f(*x)
		ret = Tensor(ctx.forward(*[_x.data for _x in x], **kwargs), requires_grad=ctx.requires_grad)
		if ctx.requires_grad: ret._ctx = ctx
		return ret	

def register(name, f):
	setattr(Tensor, name, partialmethod(f.apply, f))

class Matmul(Function):
	def __repr__(self): return "matmul"
	
	def forward(self, x, y):
		self.x, self.y = x, y
		return x @ y

	def backward(self, grad_output):
		return Tensor(grad_output @ self.y.T), Tensor(self.x.T @ grad_output)

class Sum(Function):
	def __repr__(self): return "sum"

	def forward(self, x):
		self.x = x
		return np.array(x.sum())

	def backward(self, grad_output):
		return Tensor(grad_output * np.ones_like(self.x))

class Mean(Function):
	def __repr__(self): return "mean"

	def forward(self, x):
		self.x = x
		return np.array(x.mean())

	def backward(self, grad_output):
		return Tensor(grad_output * np.ones_like(self.x) * 1/self.x.size)

class ReLU(Function):
	def __repr__(self): return "relu"

	def forward(self, x):
		self.x = x
		return np.maximum(x,0)

	def backward(self, grad_output):
		return Tensor(grad_output * (self.x >= 0))

class LogSoftmax(Function):
	def __repr__(self): return "logsoftmax"

	def forward(self, x):
		def _logsumexp(x):
			# The Log-Sum-Exp Trick: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
			c = x.max(axis=1, keepdims=True)
			return c + np.log(np.exp((x-c)).sum(axis=1, keepdims=True))

		x -= _logsumexp(x)
		#x = x - np.log(np.exp(x).sum())
		self.x = x
		return x

	def backward(self, grad_output):
		return Tensor(grad_output - np.exp(self.x) * grad_output.sum(axis=1, keepdims=True))
		return Tensor(grad_output - (np.exp(self.x) * grad_output).sum(axis=1, keepdims=True))

class Mul(Function):
	def __repr__(self): return "mul"

	def forward(self, x, y):
		self.x, self.y = x, y
		return x * y

	def backward(self, grad_output):
		return Tensor(grad_output * self.y), Tensor(grad_output * self.x)

class Add(Function):
	def __repr__(self): return "add"

	def forward(self, x, y):
		self.x, self.y = x, y
		return x + y

	def backward(self, grad_output):
		return Tensor(grad_output), Tensor(grad_output)

register("matmul", Matmul)
register("add", Add)
register("mul", Mul)
register("sum", Sum)
register("mean", Mean)
register("relu", ReLU)
register("logsoftmax", LogSoftmax)

if __name__ == "__main__":
	x = Tensor.randn(32,28*28)
	w1 = Tensor.randn(28*28,128).requires_grad_()
	w2 = Tensor.randn(128,10).requires_grad_()
	print(x)
	print(w1)
	print(w2)

	out = x.matmul(w1).relu().matmul(w2).logsoftmax()
	print(out.data)
	out.backward()

	print(w1.grad)
	print(w2.grad)

	#out = x.matmul(y)
	#print(out)
