import numpy as np
from typing import Sequence, Optional

def coverage_specificity_from_nonzero(
    nonzero: Sequence[float],
    n_cells: int,
    mass: float = 0.90,
    winsor: Optional[float] = None
) -> float:
    """
    Coverage-specificity using only strictly-positive entries.

    Parameters
    ----------
    nonzero : 1D array-like
        Strictly positive counts for this gene (zeros omitted).
        Only its length (nnz) and values matter.
    n_cells : int
        Total number of cells (including zero-expression cells).
    mass : float, default 0.90
        Target fraction of total expression to cover (0 < mass <= 1).
    winsor : float or None, default None
        Optional upper-tail winsorization level in (0, 0.5).
        Quantile is interpreted on the *full* distribution (zeros included),
        then mapped to the nonzero quantile. If the requested upper quantile
        lies within the zero mass, winsorization is skipped (no clipping).

    Returns
    -------
    float
        S_mass in [0, 1], or np.nan if there is no positive signal
        (len(nonzero)==0) or n_cells <= 0.
    """
    # Basic checks
    if n_cells <= 0:
        return np.nan

    x = np.asarray(nonzero, dtype=float)
    nnz = x.size
    if nnz == 0:
        return np.nan  # all-zero gene

    # Optional: guard against accidental zeros in 'nonzero'
    if np.any(x <= 0):
        x = x[x > 0]
        nnz = x.size
        if nnz == 0:
            return np.nan

    # ---- Optional upper-tail winsorization with zeros accounted for ----
    # We want to clip only the highest nonzero values, based on a full
    # quantile that includes zeros. Let zf = zero fraction.
    if winsor is not None and 0.0 < winsor < 0.5:
        zf = 1.0 - (nnz / float(n_cells))     # fraction of zeros in full vector
        p_full = 1.0 - winsor                 # desired full upper quantile
        if p_full > zf:                       # only meaningful if above zero mass
            # Map full-quantile to nonzero-quantile:
            p_nz = (p_full - zf) / max(1e-12, (1.0 - zf))
            p_nz = float(np.clip(p_nz, 0.0, 1.0))
            hi = np.quantile(x, p_nz)
            np.minimum(x, hi, out=x)
        # else: skip winsorization (top quantile lies inside zeros)

    tot = float(x.sum())
    if not np.isfinite(tot) or tot <= 0.0:
        return np.nan

    # Sort nonzeros descending and form cumulative sums
    xs = np.sort(x)[::-1]
    cs = np.cumsum(xs)
    target = mass * tot

    # If target is (numerically) the total, we need all nnz positives
    if target >= cs[-1] - 1e-12:
        q_star = nnz / float(n_cells)
        return 1.0 - q_star

    # Find minimal k with cs[k] >= target (0-based)
    k = int(np.searchsorted(cs, target, side='left'))

    if k == 0:
        # Target lies within the largest value
        frac_last = target / xs[0]                     # ∈ (0,1]
        q_star = (frac_last) / float(n_cells)
    else:
        # Need k full cells + a fraction of the (k+1)-th
        need = target - cs[k - 1]                      # ∈ (0, xs[k]]
        frac_last = need / xs[k]                       # ∈ (0,1]
        q_star = (k + frac_last) / float(n_cells)

    return float(1.0 - q_star)