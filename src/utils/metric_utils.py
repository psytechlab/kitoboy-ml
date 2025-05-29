from grakel import Graph
import spacy
import numpy as np

def spacy2grakel(texts: list[spacy.tokens.Doc], use_pos: bool = False, use_dep: bool = False):
    """Convert spacy doc list into graphs from grakel library.
    
    Args:
        texts: the list of the spacy docs
        use_pos: If True, use pos tags as leafes, words otherwise
        use_dep: if True, append to the node label the dependency type
    Return:
        (list): list of grakel.Graph objects
    """
    all_graphs = []
    node2id = {}
    for s in texts:
        edges = []
        node_labels = {}
        for token_id, token in enumerate(s):
            token_node = token.pos_ if use_pos else token.text
            token_dep = token.dep_
            edge = (token.head.i, token.i)
            if use_dep:
                token_node = f"{token_node}_{token_dep}"
            if node2id.get(token_node, None) is None:
                node2id[token_node] = len(node2id)
            node_labels[token.i] = node2id[token_node]
            edges.append(edge)
        gr = Graph(initialization_object=edges, node_labels=node_labels, graph_format="adjacency")
        all_graphs.append(gr)
    return all_graphs

def calculate_homogenity(pairwise_distance_matrix: np.array):
    """Calculate the homogenity of each object based on its distance from another objects.
    
    The homogenity is defined as mean of distances from one object to another.
    Calculated from pairwise distance matrix.
    
    Args:
        pairwise_distance_matrix (np.array): pairwise distance matrix n x n
    Return
        (np.array): n-shaped array of homogenity score for each object."""
    return (pairwise_distance_matrix.sum(1)-np.diag(pairwise_distance_matrix))/(pairwise_distance_matrix.shape[1]-1)
