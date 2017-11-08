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
import pickle
import os

# def parse_args():
#     '''
#     Parses the node2vec arguments.
#     '''
#     parser = argparse.ArgumentParser(description="Run node2vec.")
# 
#     parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
#                         help='Input graph path')
# 
#     parser.add_argument('--output', nargs='?', default='emb/karate.emb',
#                         help='Embeddings path')
# 
#     parser.add_argument('--dimensions', type=int, default=128,
#                         help='Number of dimensions. Default is 128.')
# 
#     parser.add_argument('--walk-length', type=int, default=80,
#                         help='Length of walk per source. Default is 80.')
# 
#     parser.add_argument('--num-walks', type=int, default=10,
#                         help='Number of walks per source. Default is 10.')
# 
#     parser.add_argument('--window-size', type=int, default=10,
#                         help='Context size for optimization. Default is 10.')
# 
#     parser.add_argument('--iter', default=1, type=int,
#                       help='Number of epochs in SGD')
# 
#     parser.add_argument('--workers', type=int, default=8,
#                         help='Number of parallel workers. Default is 8.')
# 
#     parser.add_argument('--p', type=float, default=1,
#                         help='Return hyperparameter. Default is 1.')
# 
#     parser.add_argument('--q', type=float, default=1,
#                         help='Inout hyperparameter. Default is 1.')
# 
#     parser.add_argument('--weighted', dest='weighted', action='store_true',
#                         help='Boolean specifying (un)weighted. Default is unweighted.')
#     parser.add_argument('--unweighted', dest='unweighted', action='store_false')
#     parser.set_defaults(weighted=False)
# 
#     parser.add_argument('--directed', dest='directed', action='store_true',
#                         help='Graph is (un)directed. Default is undirected.')
#     parser.add_argument('--undirected', dest='undirected', action='store_false')
#     parser.set_defaults(directed=False)
# 
#     return parser.parse_args()

# def read_graph():
#     '''
#     Reads the input network in networkx.
#     '''
#     if args.weighted:
#         G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
#     else:
#         G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
#         for edge in G.edges():
#             G[edge[0]][edge[1]]['weight'] = 1
#
#     if not args.directed:
#         G = G.to_undirected()
#
#     return G

def read_graph(conf_file, author_file, paper_file, paper_conf_file, paper_author_file):
    i = 0
    G = nx.Graph()
    confid_map_node = dict()
    authorid_map_node = dict()
    paperid_map_node = dict()
    with open(conf_file) as f:
        line = f.readline()
        while line != '':
            line = line.strip().split()
            G.add_node(i, type = 'C', name = line[1])
            confid_map_node[int(line[0])] = i
            line = f.readline()
            i = i + 1
    with open(author_file) as f:
        line = f.readline()
        while line != '':
            line = line.strip().split()
            G.add_node(i, type = 'A', name = line[1])
            authorid_map_node[int(line[0])] = i
            line = f.readline()
            i = i + 1
    with open(paper_file) as f:
        line = f.readline()
        while line != '':
            line = line.strip().split()
            G.add_node(i, type = 'P', name= ' '.join(line[1:]))
            print line
            paperid_map_node[int(line[0])] = i
            line = f.readline()
            i = i + 1
    with open(paper_conf_file) as f:
        line = f.readline()
        while line != '':
            line = line.strip().split()
            G.add_edge(paperid_map_node[int(line[0])], confid_map_node[int(line[1])])
            line = f.readline()
    with open(paper_author_file) as f:
        line = f.readline()
        while line != '':
            line = line.strip().split()
            G.add_edge(paperid_map_node[int(line[0])], authorid_map_node[int(line[1])])
            line = f.readline()
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    return (confid_map_node, authorid_map_node, paperid_map_node, G)







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
        self.linear1 = nn.Linear(2 * CONTEXT_SIZE * EMBEDDING_SIZE, 128)
        self.linear2 = nn.Linear(128, NODE_SIZE)
        self.classif1 = nn.Linear(EMBEDDING_SIZE, 64)
        self.classif2 = nn.Linear(64, 1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        clas_embed = self.embeddings(inputs[-1]).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        out = F.relu(self.classif1(clas_embed))
        out = self.classif2(out)
        clas_log_probs = F.log_softmax(out)
        return log_probs, clas_log_probs





def train_node2vec(EMBEDDING_SIZE, CONTEXT_SIZE, NODE_SIZE, random_paths, node_map_type):
    # losses = []
    loss_function = nn.NLLLoss()
    model = Node2Vec(NODE_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train_grams = []
    for random_path in random_paths:
        train_grams = train_grams + [ (random_path[i-CONTEXT_SIZE:i] + random_path[i+1: i+CONTEXT_SIZE+1], random_path[i]) for i in range(CONTEXT_SIZE, len(random_path) - CONTEXT_SIZE)]
    print train_grams[:3]
    for epoch in range(10):
        total_loss = torch.Tensor([0])
        for context, target in train_grams:
            target_class = node_map_type[context[-1]]
            # target_class = node_map_type[target]
            context_var = autograd.Variable(torch.LongTensor(context))
            model.zero_grad()
            log_probs, clas_log_probs = model(context_var)
            loss1 = loss_function(log_probs, autograd.Variable(torch.LongTensor(target)))
            loss2 = loss_function(clas_log_probs, autograd.Variable(torch.LongTensor(target_class)))
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        print total_loss




def get_node_map_type(G):
    node_map_type = dict()
    for node in list(G.nodes()):
        node_map_type[node] = G.node[node]['type']
    return node_map_type

if __name__ == '__main__':
    # nx_G = read_graph()
    if not os._exists('graph_dict.pkl'):
        conf_file = 't_id_conf.txt'
        author_file = 't_id_author.txt'
        paper_file = 't_id_paper.txt'
        paper_conf_file = 't_paper_conf.txt'
        paper_author_file = 't_paper_author.txt'
        (confid_map_node, authorid_map_node, paperid_map_node, nx_G) = read_graph(conf_file, author_file, paper_file, paper_conf_file, paper_author_file)
        graph_dict = (confid_map_node, authorid_map_node, paperid_map_node, nx_G)
        with open('graph_dict.pkl', 'w') as f:
            pickle.dump(graph_dict, f)
    else:
        with open('graph_dict.pkl') as f:
            (confid_map_node, authorid_map_node, paperid_map_node, nx_G) = pickle.load(f)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    num_walks = 10
    walk_length = 1000
    walks = G.simulate_walks(num_walks, walk_length)
    node_map_type = get_node_map_type(G.G)
    EMBEDDING_SIZE = 30
    CONTEXT_SIZE = 4
    NODE_SIZE = len(G.G.nodes())
    train_node2vec(EMBEDDING_SIZE, CONTEXT_SIZE, NODE_SIZE, walks, node_map_type)

# def main(args):
#     '''
#     Pipeline for representational learning for all nodes in a graph.
#     '''
#     nx_G = read_graph()
#     G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
#     G.preprocess_transition_probs()
#     walks = G.simulate_walks(args.num_walks, args.walk_length)
# 	node_map_type = get_node_map_type(G)
#     # learn_embeddings(walks)
#     EMBEDDING_SIZE = 300
#     CONTEXT_SIZE = 4
#     NODE_SIZE = len(nx_G.nodes)
#     train_node2vec(EMBEDDING_SIZE, CONTEXT_SIZE, NODE_SIZE, walks, node_map_type)

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)
