import numpy as np
import scipy.io as io
import networkx as nx


def pagerank(mat_file: str = '../gr_900_900_crg.mtx', max_iter: int = 1, epsilon: float = 1.e-14) -> None:
    A = io.mmread(mat_file)
    G = nx.from_scipy_sparse_matrix(A)
    pr = nx.pagerank(G, alpha=0.85, max_iter=1, tol=100)
    print(list(pr.values()))


if __name__ == '__main__':
    pagerank()
