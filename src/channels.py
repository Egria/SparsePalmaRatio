import pandas as pd
import scipy.sparse as sp
from .select_genes import select_genes
from .activate import make_binary
from .graph.make_graph import make_graph
from .graph.graph_mix import mix_knn_graphs
from .parameters import Parameters
from .tsne_plot import tsne_from_feature

def make_channel_graphs(params: Parameters, gene_stats:pd.DataFrame, matrix:sp.csr_matrix,
                        genes_f, labels, cells_f, cells, b5=None, b4=None, b3=None, b2=None, b1=None, bands=None,
                        band_weights = None):
    if b5 is None: b5 = {"fano":0.6, "gini":0.2, "theil":0.2}
    if b4 is None: b4 = {"gini":0.5, "theil":0.3, "fano":0.2}
    if b3 is None: b3 = {"gini":0.45, "palma":0.35, "theil":0.2}
    if b2 is None: b2 = {"palma":0.6, "theil":0.25, "gini":0.15}
    if b1 is None: b1 = {"palma":0.9, "theil":0.1}



    if bands is None:
        bands = [
            ("50-30", b5, 0.0, params.gene_nfeatures),
            ("30-10", b4, 0.0, params.gene_nfeatures),
            ("10-3", b3, 0.0, params.gene_nfeatures ),
            ("3-1", b2, 0.0, params.gene_nfeatures ),
            ("1-0.1", b1, 3.5, params.gene_nfeatures //4 )
        ]
    band_graphs = []
    if band_weights is None: band_weights = [0.18, 0.2, 0.22, 0.22, 0.18] #[0.1, 0.15, 0.2, 0.25, 0.3]



    band_genes = []
    for band in bands:
        print(band[0])
        qualifiers, weighted_ranked_p = select_genes(params, gene_stats, band[1], band[3], band[2])
        binarized, zero_cells = make_binary(params, matrix, genes_f, qualifiers)
        print(zero_cells.sum())
        graph = make_graph(params, binarized)
        #emb_df = tsne_from_feature(params, binarized, labels, cells_f, cells, output_suffix=band[0])
        band_graphs.append(graph)
        band_genes.append(qualifiers)
    G = mix_knn_graphs(band_graphs, band_weights)
    return G, band_genes

