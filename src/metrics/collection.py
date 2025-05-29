"""A collection of metrics from [this paper](https://arxiv.org/pdf/2101.06030).

To compute the pairwise distance matrix, you can use sklearn.metrics.pairwise_distances. 
"""
import numpy as np
from scipy.spatial.distance import cdist
from src.utils.mst import compute_minimum_spanning_tree

def compute_remote_clique(pairwise_distances: np.array) -> float:
    """Compute remote clique metric.
    
    Average of mean pairwise distances. While commonly used in crowd 
    ideation studies, it is insensitive to highly clustered points.

    Args:
        pairwise_distances (np.array): A precalculated pairwise distance.

    Returns:
        float: The computed metric value.
    """
    return np.sum(pairwise_distances) / (pairwise_distances.shape[0] ** 2)

def compute_chamfer_distance(pairwise_distances: np.array):
    """Compute chamfer distance.
    
    Average of minimum pairwise distances. Chamfer distance (or 
    Remote-pseudoforest) measures the distance to the nearest
    neighbor. However, it is biased when points are clustered.

    Args:
        pairwise_distances (np.array): A precalculated pairwise distance.

    Returns:
        float: The computed metric value.
    """
    mask = np.ones(pairwise_distances.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    mins = np.min(pairwise_distances[mask].reshape(-1,pairwise_distances.shape[0]-1), axis=1)
    return mins.mean()

def compute_mst_dispersion(pairwise_distances: np.array):
    """Compute minimum spanning tree dispersion.
    
    Popular in ecology research as functional diversity, and called
    Remote-tree or Remote-MST, this learns a minimum spanning
    tree (MST) of the points, and calculates the sum of edge weights.

    Args:
        pairwise_distances (np.array): A precalculated pairwise distance.

    Returns:
        float: The computed metric value.
    """
    mst = compute_minimum_spanning_tree(pairwise_distances)
    return sum([ pairwise_distances[x[0], x[1]] for x in mst]) / len(mst)
    
def compute_sparseness(pairwise_distances: np.array, X: np.array):
    """Compute sparseness metric.
    
    Sparsity of points positioned around the medoid 
    ($argmin_{x_i}=\{\sim^N_{j=1}d(\bold{x}_i,\bold{x}_j)\}$)
    If points cluster around the medoid, then this metric will 
    be small (i.e., not sparse).

    Args:
        pairwise_distances (np.array): A precalculated pairwise distance.
        X (np.array): A source representation vectors. 

    Returns:
        float: The computed metric value.
    """
    medoid = X[np.argmin(pairwise_distances.sum(axis=0), axis=0)]
    return cdist(medoid.reshape(1, -1), X).mean()

def compute_span_metric(X: np.array):
    """Compute span metric.
    
    P-th percentile distance to centroid (\sum^N_{i=1}=)

    Args:
        X (np.array): A source representation vectors. 

    Returns:
        float: The computed metric value.
    """
    centroid = X.mean(axis=0)
    return np.percentile(cdist(centroid.reshape(1, -1), X), 90)