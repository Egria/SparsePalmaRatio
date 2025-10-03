import pandas as pd

from .parameters import Parameters
from .preprocess import preprocess
from .filter import filter_counts
from .calc_stat import calc_gene_stats
from .detrend import detrend
from .select_genes import select_genes
from .binarize import make_binary
from .make_graph import make_graph
from .generate_cluster import generate_clusters
from .comparison import compare_clusters_filtered

def cell_identify(config_filename:str):
    params = Parameters(config_filename)
    matrix, cells, genes, labels = preprocess(params)
    matrix_f, genes_f, cells_f = filter_counts(params, matrix, genes, cells, False)
    log2max, gini, palma = calc_gene_stats(params, matrix_f, genes_f)
    gini_norm, palma_norm = detrend(params, genes_f, log2max, gini, palma)
    if params.method == 'gini':
        stat = gini_norm
    elif params.method == 'palma':
        stat = palma_norm
    else:
        raise NotImplementedError("METRICS NOT IMPLEMENTED")
    qualifiers, stat_q = select_genes(params, genes_f, stat)
    binarized, zero_cells = make_binary(params, matrix_f, genes_f, qualifiers)
    graph = make_graph(params, binarized)
    labels_f = generate_clusters(params, graph, cells_f)
    result = pd.DataFrame({"label":labels_f}, index=cells_f)
    tab, ari, nmi = compare_clusters_filtered(params, labels_f, labels, cells_f, cells)
    #print(result["label"].value_counts())



