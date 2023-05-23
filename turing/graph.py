from graphviz import Digraph

# Tensor.deepwalk() -> nodes

def trace(root):
	nodes, edges = set(), set()
	def build(v):
		if v not in nodes:
			nodes.add(v)
			for parent in v._ctx.parents:
				edges.add((parent, v))

	build(root)
	return nodes, edges

def draw(root):
	dot = Digraph(format="svg", graph_attr={"rankdir": "LR"}) # left -> right

	nodes, edges = trace(root)
	for n in nodes:
		uid = str(id(n))
		dot.node(name=uid, label=str(n._ctx), shape="record")
		# ... 

	#for n1, n2 in edges:
#		dot.edge(str(id(n1)), str(id(n2)) + 

	return dot
