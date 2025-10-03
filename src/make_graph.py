from .graph.jaccard_radius_graph import jaccard_radius_graph_blockwise
from .parameters import Parameters
import scipy.sparse as sp

def make_graph(params: Parameters, B: sp.csr_matrix):


    obj = jaccard_radius_graph_blockwise(B, eps=params.dbscan_eps)
    return obj