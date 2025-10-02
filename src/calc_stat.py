import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from .parameters import Parameters
from .metrics.gini_index import gini_index_sparse_exact
from .metrics.palma_ratio import palma_ratio_from_sparse_nonzeros

def calc_gene_stats(
    params: Parameters,
    X_f: sp.csr_matrix,
    genes_f,  # pandas.Index or Series
    gini_unbiased: bool = True,
    log2_eps: float = 0.0,      # set 0.1 if you want the earlier (+0.1) behavior
):
    """
    EXACT per-gene stats on CSR (genes x cells):
      - gini (your exact formula, zero-aware)
      - palma_a (Palma with alpha), exact
      - log2max = log2(max_count + log2_eps) using integer counts
    """
    if not sp.isspmatrix_csr(X_f):
        X_f = X_f.tocsr(copy=False)
    X_f.eliminate_zeros()

    n_genes, n_cells = X_f.shape
    indptr = X_f.indptr
    data = X_f.data

    gini = np.empty(n_genes, dtype=np.float64)
    palma_a = np.empty(n_genes, dtype=np.float64)
    max_counts = np.zeros(n_genes, dtype=np.int64)

    for i in range(n_genes):
        s, e = indptr[i], indptr[i + 1]
        v = data[s:e]  # strictly positive integer counts for gene i
        if e > s:
            # max from integers
            max_counts[i] = int(v.max())
            # exact Gini and Palma from sparse nonzeros + n_cells
            gini[i] = gini_index_sparse_exact(v, n_cells, unbiased=gini_unbiased)
            palma_a[i] = palma_ratio_from_sparse_nonzeros(
                v, n_cells,
                upper=params.palma_upper, lower=params.palma_lower,
                alpha=params.palma_alpha, winsor=params.palma_winsor
            )
        else:
            raise ZeroDivisionError(f"GENE WITH NO EXPRESSIONS: {genes_f[i]}")

    # log2 of integer max counts (optionally with a small eps)
    log2max = np.log2(max_counts.astype(np.float64) + float(log2_eps))

    csv_path = os.path.join(params.output_folder,"gene_stats.csv")
    pd.DataFrame({"gini":gini, "palma":palma_a, "log2max": log2max}, index=genes_f).to_csv(csv_path,
                                                                                           index=True, header=True)
    return log2max, gini, palma_a