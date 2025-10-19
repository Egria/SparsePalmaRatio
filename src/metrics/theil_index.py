import numpy as np

def theil_index_from_sparse_nonzeros(
    nonzero: np.ndarray,
    n_cells: int
) -> float:
    """
    Exact Theil index T for one gene over ALL cells (zeros included).

    Definition (natural log):  T = (1/n) * Σ_i (x_i/μ) * ln(x_i/μ),
    with the convention 0 * ln(0) := 0, so zeros contribute 0.

    Parameters
    ----------
    nonzero : 1D array-like of floats/ints
        Strictly positive counts of this gene (zeros omitted).
    n_cells : int
        Total number of cells (including zeros).
    log_base : float
        Base of the logarithm (default: e). For base b, we compute ln(·)/ln(b).

    Returns
    -------
    T : float
        Theil index (≥ 0). Raises ZeroDivisionError if the mean is 0.
    """
    v = np.asarray(nonzero, dtype=np.float64)
    n = int(n_cells)
    if n <= 0:
        return np.nan

    S = v.sum()
    mu = S / n
    if mu <= 0.0:
        raise ZeroDivisionError("Mean is zero for Theil index (all-zero gene).")

    # Only non-zeros contribute; zeros add 0 exactly.
    ratio = v / mu
    # use natural log then change of base if needed
    ln_ratio = np.log(ratio)

    T = (ratio * ln_ratio).sum() / n
    # numerical guard: tiny negatives to zero
    return float(max(T, 0.0))