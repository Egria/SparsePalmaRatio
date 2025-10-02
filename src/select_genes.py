import numpy as np
import pandas as pd
from .parameters import Parameters

def select_genes(
    params : Parameters,
    genes,
    scores: np.ndarray
):
    """
    Simplified single-metric selector (no NaNs).
    Returns a ranked pd.Index of selected genes.
      - standard="top": pick top-N (or top-ratio) by score (descending).
      - standard="pval": pick genes with one-sided right-tail normal p < alpha,
                         ranked by ascending p-value.
    """
    standard = params.select_standard
    top = params.gene_topcut
    alpha = params.pval_threshold
    # normalize inputs
    if isinstance(genes, pd.Series):
        gene_idx = pd.Index(genes.to_numpy())
    elif isinstance(genes, pd.Index):
        gene_idx = genes
    else:
        gene_idx = pd.Index(np.asarray(genes))
    s = np.asarray(scores, dtype=float)
    n = gene_idx.size
    if s.shape[0] != n:
        raise ValueError("genes and scores must have the same length")

    if standard == "top":
        # k from ratio or absolute N
        k = int(np.ceil(top * n)) if (isinstance(top, float) and 0 < top <= 1.0) else int(top)
        k = max(1, min(n, k))
        idx_k = np.argpartition(s, -k)[-k:]               # top-k (unordered)
        idx_sorted = idx_k[np.argsort(s[idx_k])[::-1]]    # rank by score (desc)
        return gene_idx.take(idx_sorted), s[idx_sorted]

    elif standard == "pval":
        # one-sided right-tail under N(mu, sigma^2)
        mu = s.mean()
        sigma = s.std(ddof=1)
        if sigma <= 0:
            return pd.Index([])  # no variation → no right outliers
        z = (s - mu) / sigma
        # p = 1 - Phi(z) = 0.5 * erfc(z / sqrt(2))
        p = 0.5 * np.erfc(z / np.sqrt(2.0))
        mask = (s > mu) & (p < alpha)                     # right outliers only
        sel = np.flatnonzero(mask)
        if sel.size == 0:
            raise ValueError("NO GENES SELECTED")
            return pd.Index([])
        order = np.argsort(p[sel])                        # rank by p-value (asc)
        return gene_idx.take(sel[order]), s[sel[order]]

    else:
        raise ValueError("standard must be 'top' or 'pval'")