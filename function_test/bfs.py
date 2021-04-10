import numpy as np
from scipy.io import mmread
import networkx as nx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file', metavar='F', type=str)
parser.add_argument('source_vertex', metavar='sv', type=int)
args = parser.parse_args()

mat = mmread(args.file)
# import code
# code.interact(local=locals())
G = nx.from_scipy_sparse_matrix(mat)
predecessors = [_[1] for _ in sorted(nx.bfs_predecessors(G, args.source_vertex))]
for _ in predecessors:
    print(_)