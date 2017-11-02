'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.save_word2vec_format(args.output)
	return

class Node2Vec(nn.Module):
    def __init__(self, NODE_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE):
        super(Node2Vec, self).__init__()
        self.embeddings = nn.Embedding(NODE_SIZE, EMBEDDING_SIZE)
        self.linear1 = nn.Linear(CONTEXT_SIZE * EMBEDDING_SIZE, 128)
        self.linear2 = nn.Linear(128, NODE_SIZE)
		self.classif1 = nn.Linear(EMBEDDING_SIZE, 64)
		self.classif2 = nn.Linear(64, 1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs[:-1]).view((1, -1))
		clas_embed = self.embeddings(inputs[-1]).view((1,-1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
		out = F.relu(self.classif1(clas_embed))
		out = self.classif2(out)
		clas_log_probs = F.log_softmax(out)
        return log_probs, clas_log_probs





def train_node2vec(EMBEDDING_SIZE, CONTEXT_SIZE, NODE_SIZE, Random_path):
	losses = []
	loss_function = nn.NLLLoss()
	model = Node2Vec(NODE_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)
	optimizer = optim.SGD(model.parameters(), lr=0.001)
	for epoch in range(10):
		total_loss = torch.Tensor([0])
		for context, target in Random_path:
			context_var = autograd.Variable(torch.LongTensor(context))
			model.zero_grad()
			log_probs, clas_log_probs = model(context_var)
			loss1 = loss_function(log_probs, autograd.Variable(torch.LongTensor(target)))
			loss2 = loss_function(clas_log_probs, autograd.Variable(torch.LongTensor(target_class)))
			loss = loss1 + loss2
			loss.backward()







def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph()
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)
