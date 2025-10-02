from .parameters import Parameters
from .preprocess import preprocess
from .filter import filter_counts
from .calc_stat import calc_gene_stats
from .detrend import detrend
from .select_genes import select_genes

def cell_identify(config_filename:str):
    params = Parameters(config_filename)
    matrix, cells, genes, labels = preprocess(params)
    matrix_f, genes_f, cells_f = filter_counts(params, matrix, genes, cells, False)
    log2max, gini, palma = calc_gene_stats(params, matrix_f, genes_f)
    gini_norm, palma_norm = detrend(params, genes_f, log2max, gini, palma)
    qualifiers, gini_q = select_genes(params, genes_f, gini_norm)



