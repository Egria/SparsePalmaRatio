import numpy as np
from scipy.optimize import least_squares

# ---- helpers ----
def _exp_plateau(theta, x):
    a, b, g = theta
    return a + b * np.exp(-g * x)

def _equal_count_bins(x, y, w, n_bins):
    n = x.size
    n_bins = int(np.clip(n_bins, 20, min(400, n // 50 if n >= 2000 else n // 10 or 1)))
    o = np.argsort(x, kind='mergesort')
    xs, ys, ws = x[o], y[o], w[o]
    idx = np.linspace(0, n, n_bins + 1, dtype=int)
    bx, by, bw = [], [], []
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a < 5:
            continue
        xb, yb, wb = xs[a:b], ys[a:b], ws[a:b]
        bx.append(np.median(xb))
        by.append(np.median(yb))
        bw.append(wb.sum())
    if not bx:  # fallback
        return xs, ys, ws
    return np.asarray(bx), np.asarray(by), np.asarray(bw)

def _init_guess(bx, by):
    # right plateau ~ median of top-x bins; b0 ~ left-minus-right; g0 ~ gentle decay
    hi = np.percentile(bx, 90) if bx.size >= 10 else bx.max()
    lo = np.percentile(bx, 10) if bx.size >= 10 else bx.min()
    a0 = np.median(by[bx >= hi]) if np.any(bx >= hi) else by[-1]
    b0 = max((np.median(by[bx <= lo]) if np.any(bx <= lo) else by[0]) - a0, 1e-6)
    g0 = 2.0 / max(hi - lo, 1e-3)
    return np.array([a0, b0, g0], float)

def _fit_exp_plateau(x, y, w=None, n_bins=200, right_cap_q=0.995,
                     robust_loss='soft_l1', f_scale=1.0):
    n = x.size
    if w is None:
        w = np.ones(n, float)
    # Cap x for fitting (prevents right-edge extrapolation from collapsing)
    x_cap = np.quantile(x, right_cap_q)
    xfit = np.minimum(x, x_cap)

    # Bin to stabilize, then fit on binned medians with counts as weights
    bx, by, bw = _equal_count_bins(xfit, y, w, n_bins=n_bins)
    theta0 = _init_guess(bx, by)

    def resid(theta):
        return np.sqrt(bw) * (_exp_plateau(theta, bx) - by)

    bounds = (np.array([-np.inf, 0.0, 0.0]),  # a free, b>=0, g>=0
              np.array([ np.inf, np.inf, np.inf]))
    res = least_squares(resid, theta0, bounds=bounds,
                        loss=robust_loss, f_scale=f_scale,
                        method='trf', max_nfev=2000)
    a, b, g = res.x

    # Predictor function; clamp to x_cap when evaluating
    def f(z):
        z = np.asarray(z, float)
        zc = np.minimum(z, x_cap)
        return _exp_plateau((a, b, g), zc)

    return f, dict(alpha=float(a), beta=float(b), gamma=float(g),
                   x_cap=float(x_cap), success=bool(res.success), nfev=int(res.nfev))

# ---- drop-in API (two-pass) ----
def leastsq_linear_detrending(gini: np.ndarray,
                              log2max: np.ndarray,
                              outlier: float = 0.75,
                              span: float = 0.9):
    """
    Two-pass detrending using y(x) = a + b*exp(-g x) instead of LOESS.
    Arguments kept for API compat; 'span' is ignored.

    Returns
    -------
    r2 : np.ndarray
        Second-pass residuals y - yhat(x) at all points.
    """
    assert len(gini) == len(log2max)
    x = np.asarray(log2max, float)
    y = np.asarray(gini, float)

    # ---- 1) first fit on all points ----
    f1, _ = _fit_exp_plateau(x, y)

    r1 = y - f1(x)

    # ---- 2) inlier mask via upper-quantile of positive residuals (same rule as your LOESS code) ----
    pos = r1[r1 > 0]
    thresh = np.quantile(pos, outlier) if pos.size else np.inf
    inlier = r1 < thresh

    # ---- 3) second fit on inliers only ----
    f2, _ = _fit_exp_plateau(x[inlier], y[inlier])

    # ---- 4) final residuals (no interpolation needed; model is analytic) ----
    r2 = y - f2(x)
    return r2