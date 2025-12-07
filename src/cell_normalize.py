import numpy as np
import scipy.sparse as sp
from typing import Tuple, Union

def cpm_normalize_csr(
    matrix_f: sp.csr_matrix,
    target_sum: float = 1_000_000.0,   # CPM
    copy: bool = True,
    return_size_factors: bool = False
) -> Union[sp.csr_matrix, Tuple[sp.csr_matrix, np.ndarray]]:
    """
    Column-wise (per-cell) CPM normalization for a (genes x cells) CSR sparse matrix.

    Parameters
    ----------
    matrix_f : sp.csr_matrix
        Raw counts, shape (ngenes, ncells). Columns are cells.
    target_sum : float, default 1e6
        Desired total per cell after normalization. Use 1e4 for CP10K, etc.
    copy : bool, default True
        If True, operate on a copy; otherwise normalize in-place.
    return_size_factors : bool, default False
        If True, also return the per-cell scaling factors.

    Returns
    -------
    X_norm : sp.csr_matrix
        CPM-normalized matrix in CSR format (same shape).
    size_factors : np.ndarray, optional
        Vector of per-cell scale factors (length = ncells), returned
        if return_size_factors is True.

    Notes
    -----
    - Cells with zero total counts are left as all zeros (scale factor = 0).
    - Works in O(nnz) time by scaling entries with column-wise factors.
    """
    if not sp.isspmatrix_csr(matrix_f):
        raise TypeError("matrix_f must be a scipy.sparse.csr_matrix with shape (ngenes, ncells)")

    X = matrix_f.copy() if copy else matrix_f

    # Per-cell (column) sums
    col_sums = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)

    # Compute scale factors; zero for empty cells
    scale = np.divide(
        target_sum, col_sums,
        out=np.zeros_like(col_sums, dtype=np.float64),
        where=col_sums > 0
    )

    # Scale columns in-place: data[k] belongs to column X.indices[k]
    X.data = X.data.astype(np.float64, copy=False)
    X.data *= scale[X.indices]

    return (X, scale) if return_size_factors else X