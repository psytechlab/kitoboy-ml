"""https://peekaboo-vision.blogspot.com/2012/02/simplistic-minimum-spanning-tree-in.html"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import matplotlib.pyplot as plt


def compute_minimum_spanning_tree(X, copy_X=True):
    """Compute minimum spanning tree.
    
    It's a Prim's algorithm. Because it uses the adjency matrix,
    the complexity is |V^2|, so it's better not using it with the X
    more the 2-3k dim.
    
    Args:
        X (np.array): Matrix of edge weights of fully connected graph
        copy_X (bool): If true, operate on copy of the input.
    
    Return:
        np.array: 2d edge metrix that form mst.
    """
    if copy_X:
        X = X.copy()

    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []
    
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf
    
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                       
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf                                                                     
        num_visited += 1
    return np.vstack(spanning_edges)
