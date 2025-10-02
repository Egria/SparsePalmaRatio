import numpy as np
import scipy.sparse as sp
from sklearn.isotonic import IsotonicRegression

def detrend_palma_2pass_isotonic(
    x,                    # np.ndarray, log2maxs
    stat,                 # np.ndarray, raw Palma (or any positive stat)
    mad_eps=1e-12,
    mad_topcut=0.01,      # keep r1 <= quantile(1 - mad_topcut)
    mad_nbins=401         # sliding window size (odd recommended)
):
    """
    Simplified (no-NaN) rewrite of your Palma detrending:
      - y = log(max(stat, mad_eps))
      - 1st isotonic fit -> r1
      - keep <= upper-quantile outliers -> 2nd isotonic fit -> r
      - local kernel weighted MAD over x-sorted window -> s
      - z = r / s
    Returns
      r, z  (np.ndarray)
    """
    # ---- inputs as numpy ----
    x = np.asarray(x, dtype=float)
    y = np.log(np.maximum(np.asarray(stat, dtype=float), float(mad_eps)))

    # ---- two-pass isotonic baseline (decreasing) ----
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    yfit1 = iso.fit_transform(x, y)
    r1 = y - yfit1

    cutoff = np.quantile(r1, 1.0 - mad_topcut)
    keep = (r1 <= cutoff)
    # assert at least two points (as in your template)
    assert keep.sum() >= 2, "Not enough inliers for second isotonic fit."

    iso2 = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso2.fit(x[keep], y[keep])
    yhat = iso2.predict(x)
    r = y - yhat

    # ---- weighted median / MAD helpers (no NaNs) ----
    def _wmedian(v, w):
        o = np.argsort(v, kind="mergesort")
        v, w = v[o], w[o]
        cw = np.cumsum(w)
        return v[np.searchsorted(cw, 0.5 * cw[-1], side="left")]

    def _wmad(v, w, eps=1e-12):
        m = _wmedian(v, w)
        a = np.abs(v - m)
        mad = 1.4826 * _wmedian(a, w)
        if not np.isfinite(mad) or mad < eps:
            # Weighted IQR fallback if MAD collapses
            o = np.argsort(v, kind="mergesort")
            v, w = v[o], w[o]
            cw = np.cumsum(w); cw /= cw[-1]
            q25 = np.interp(0.25, cw, v)
            q75 = np.interp(0.75, cw, v)
            mad = max(0.7413 * (q75 - q25), eps)
        return mad

    # ---- local kernel MAD over x (triangular kernel in a sliding window) ----
    def _local_kernel_mad(r, x, k):
        n = r.size
        k = int(max(3, min(k, n)))
        if k % 2 == 0:
            k -= 1
        half = k // 2

        order = np.argsort(x, kind="mergesort")
        xs, rs = x[order], r[order]
        s_sorted = np.empty(n, float)

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            xr = xs[lo:hi]
            rr = rs[lo:hi]
            h = xr[-1] - xr[0]
            if h <= 0:
                w = np.ones_like(xr)
            else:
                w = 1.0 - np.abs(xr - xs[i]) / h
                w[w < 0] = 0.0
            s_sorted[i] = _wmad(rr, w, eps=mad_eps)

        s = np.empty_like(s_sorted)
        s[order] = s_sorted
        return s

    s = _local_kernel_mad(r, x, k=mad_nbins)
    z = r / s
    return r, z