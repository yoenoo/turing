import numpy as np
from functools import partialmethod

class Function:
	def __init__(self, *tensors):
		self.parents = list(tensors)
		self.saved_tensors = []

	def __repr__(self): pass
	def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {self.__class__}")
	def backward(self, *args, **kwargs): raise NotImplementedError(f"backward not implemented for {self.__class__}")

	def save_for_backward(self, *x):
		self.saved_tensors.extend(x)

	def apply(self, f, *x):
		# self = Tensor
		ctx = f(*x)
		ctx.parents.append(self)
		requires_grad = any(x.requires_grad for x in ctx.parents)
		ret = Tensor(f.forward(ctx, self.data, *[t.data for t in x]), requires_grad)
		#ret = Tensor(f.forward(ctx, *[t.data for t in x]))
		ret._ctx = ctx
		return ret

class Matmul(Function):
	def __repr__(self): return "matmul"

	@staticmethod
	def forward(ctx, input, weight):
		ctx.save_for_backward(input, weight)
		return input.dot(weight)

	@staticmethod
	def backward(ctx, grad_output):
		input, weight = ctx.saved_tensors
		grad_input = grad_output.dot(weight.T)
		grad_weight = input.T.dot(grad_output)
		#print(input.shape, weight.shape, grad_output.shape)
		#print(grad_input.shape, grad_weight.shape)
		
		"""
		print(input.shape, weight.shape, grad_output.shape)
		grad_input = weight.T.dot(grad_output)
		grad_weight = input.dot(grad_output.T)
		print(grad_input.shape, grad_weight.shape)
		"""
		return Tensor(grad_weight), Tensor(grad_input) 

class Sum(Function):
	def __repr__(self): return "sum"

	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		return np.array([input.sum()])

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		return Tensor(grad_output * np.ones_like(input))

class Tensor:
	def __init__(self, data, requires_grad = False):
		self.data = np.array(data) if isinstance(data, list) else data
		self.grad = None
		self.requires_grad = requires_grad
		self._ctx = None # instantiation of Function

	@property
	def shape(self):
		return self.data.shape

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
		assert self.shape == (1,)
		self.grad = Tensor([1], requires_grad=False) # implicit gradient

		for t0 in reversed(self.deepwalk()):
			#print(self)
			#print(t0._ctx)
			#print(t0._ctx.parents)

			if not any(x.requires_grad for x in t0._ctx.parents):
				continue

			assert t0.grad is not None
			grads = t0._ctx.backward(t0._ctx, t0.grad.data) ## TODO: fix -> redundant calls to ctx
			grads = [grads] if len(t0._ctx.parents) == 1 else grads
			for t, g in zip(t0._ctx.parents, grads):
				# print("parents:", t, g, t.requires_grad)
				# print(t.data, g.data)
				if g is not None and t.requires_grad:
					assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
					t.grad = g if t.grad is None else t.grad + g

			print()






	"""
	def backward(self, allow_fill=True):
		if self._ctx is None: return

		if self.grad is None and allow_fill: ### NOT UNDERSTAND
			if self.data.shape != (1,):
				raise RuntimeError("Does not suport backward on multidimentional array")
			self.grad = np.array([[1]]) #np.ones_like(self.data)

		assert self.grad is not None ### NOT UNDERSTAND
		grads = self._ctx.backward(self._ctx, self.grad)
		grads = [grads] if len(self._ctx.parents) == 1 else grads
		for t, g in zip(self._ctx.parents, grads):
			if not t.requires_grad:
				continue
			print(t.data)
			print(g)
			print(t.requires_grad)
			if g.shape != t.data.shape:
				raise RuntimeError(f"grad shape must match tensor shape: {g.shape!r} != {t.data.shape!r}")
		
			t.grad = g # TODO: should be +=
			t.backward(allow_fill=False)
	"""


def register(name, f):
	setattr(Tensor, name, partialmethod(f.apply, f))

register("matmul", Matmul)
register("sum", Sum)
