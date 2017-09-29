import numpy as np
import networkx as nx
import random


class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def node2vec_hin_walk(self, path, type, start_node, type_weight):
		'''
		Simulate a random walk in HIN from start node according to path
		:param path: meta path such as 'APA' or 'APCPA'
		:param type: the type name of the node. node[type] = 'A'
		:param start_node:
		:param type_weight: the type's weight name of the node. node[type_weight] = 1.0
		:return: a walk path
		'''
		G = self.G
		if G.nodes[start_node][type] != path[0]:
			print 'the start node s type is not consistent with path0 type.'
			return False
		walk = [start_node]

		walk_length = len(path)
		walk_step = 1
		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = []
			cur_nbrs_weight = []
			# print cur
			for nbr_node in sorted(G.neighbors(cur)):
				if G.nodes[nbr_node][type] == path[walk_step]:
					cur_nbrs.append(nbr_node)
					cur_nbrs_weight.append(G.nodes[nbr_node][type_weight])
			if len(cur_nbrs) == 0:
				print "can't find the type path"
				return False
			# print cur_nbrs
			# print cur_nbrs_weight
			next_node = np.random.choice(cur_nbrs, 1, replace = False, p = cur_nbrs_weight)[0]
			walk.append(next_node)

		return walk





	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print 'Walk iteration:'
		for walk_iter in range(num_walks):
			print str(walk_iter+1), '/', str(num_walks)
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def simulate_hin_walks(self, num_walks, path, type, type_weight):
		'''
		Repeatedly simulate random walks from each node in HIN according to path.
		:param num_walks: num of walks
		:param path: meta path such as 'APA' or 'APCPA'
		:param type: the type name of the node. node[type] = 'A'
		:param type_weight: the type's weight name of the node. node[type_weight] = 1.0
		:return: a list of walk
		'''
		G = self.G
		walks = []
		start_nodes = []
		for s_node in list(G.nodes()):
			if G.nodes[s_node][type] == path[0]:
				start_nodes.append(s_node)
		if len(start_nodes) == 0:
			print 'can not find this type node'
			return False
		print 'Walk iteration:'
		for walk_iter in range(num_walks):
			print str(walk_iter+1), '/' , str(num_walks)
			random.shuffle(start_nodes)
			for start_node in start_nodes:
				walk = self.node2vec_hin_walk(path, type, start_node, type_weight)
				if walk != False:
					walks.append(walk)
		return walks



	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]