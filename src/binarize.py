import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
from .parameters import Parameters

def _rawcounts_cutoff_rows(X_rows_csr: sp.csr_matrix, gamma: float) -> int:
    """
    Exact per-gene threshold selection on *just the provided rows* (genes x cells, CSR).
    Returns one integer cutoff using your Gamma heuristic.
    """
    n_cells = X_rows_csr.shape[1]
    indptr, data = X_rows_csr.indptr, X_rows_csr.data

    bc_low, bc_high = [], []
    for r in range(X_rows_csr.shape[0]):
        s, e = indptr[r], indptr[r + 1]
        v = data[s:e].astype(np.int64, copy=False)
        m = e - s
        z = n_cells - m  # zeros

        if m > 0:
            u, freq_pos = np.unique(v, return_counts=True)  # ascending
            c = np.concatenate([np.array([0], dtype=np.int64), u])
            f = np.concatenate([np.array([z], dtype=np.int64), freq_pos])
        else:
            c = np.array([0], dtype=np.int64)
            f = np.array([z], dtype=np.int64)

        denom = int(v.sum())
        if denom == 0:
            csum = np.zeros_like(c, dtype=float)
            warnings.warn("Gene used for cutoff has all zero counts.")
        else:
            contrib = (c * f).astype(np.int64, copy=False)
            tail_ge = np.cumsum(contrib[::-1], dtype=np.int64)[::-1]
            csum = tail_ge / float(denom)

        order_desc = np.argsort(c)[::-1]
        c_desc = c[order_desc]
        csum_desc = csum[order_desc]

        hits = np.where(csum_desc > gamma)[0]
        n_idx = int(hits[0]) if hits.size else 0
        n_idx = max(2, n_idx)
        if n_idx >= len(c_desc) - 1:
            n_idx = max(0, len(c_desc) - 2)

        hi = float(c_desc[n_idx]) if len(c_desc) else 0.0
        lo = float(c_desc[n_idx + 1]) if (len(c_desc) >= 2) else hi

        bc_high.append(hi)
        bc_low.append(lo)

    bc_med = 0.5 * (np.asarray(bc_high) + np.asarray(bc_low))
    top_n_gene = max(int(len(bc_med) * 0.10), 10)
    cutoff = int(np.floor(np.mean(bc_med[:top_n_gene])))
    return cutoff

def make_binary(
    params: Parameters,
    X_filtered: sp.csr_matrix,
    genes_filtered,                 # Index/Series/array aligned to rows of X_filtered
    genes_qualified                 # Index/Series/array (~50–200)
):
    """
    Memory-efficient:
      - Never binarizes all genes. Only slices the qualified genes first.
      - Computes cutoff on the selected rows only, then binarizes just those rows.
      - Returns:
          B: (cells x features) CSR binary
          obj: dense distance (np.ndarray or memmap) or sparse ε-graph (CSR)
          meta: {'cutoff', 'feature_names', 'zero_cells'}
    """
    if not sp.isspmatrix_csr(X_filtered):
        X_filtered = X_filtered.tocsr(copy=False)
    X_filtered.eliminate_zeros()

    gf = pd.Index(genes_filtered) if not isinstance(genes_filtered, pd.Index) else genes_filtered
    gq = pd.Index(genes_qualified) if not isinstance(genes_qualified, pd.Index) else genes_qualified


    # order-preserving indexer in the order of genes_qualified
    idx = gf.get_indexer(gq)
    keep = idx >= 0
    if not np.any(keep):
        raise ValueError("None of genes_qualified are present in genes_filtered.")
    idx_sel = idx[keep]
    feat_names = gq[keep]  # preserve caller's order

    # --- slice rows FIRST (only selected genes) ---
    X_sel = X_filtered[idx_sel, :]            # (n_feat x n_cells), new object; no full copy

    # --- cutoff on selected rows only ---
    cutoff = _rawcounts_cutoff_rows(X_sel, gamma=params.jaccard_gamma)

    # --- binarize ONLY the selected rows (>= cutoff) ---
    # safe to modify in place: X_sel is a slice (new object)
    X_sel.data = (X_sel.data >= cutoff).astype(np.int8, copy=False)
    X_sel.eliminate_zeros()

    # --- build B = (cells x features) ---
    B = X_sel.T.tocsr(copy=False)             # cells x selected-features
    zero_cells = (B.getnnz(axis=1) == 0)

    return B, zero_cells