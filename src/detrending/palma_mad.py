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

def _monotone_tie_jitter(x, eps=1e-9):
    x = np.asarray(x, float).copy()
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    # boundaries of equal-x runs
    b = np.r_[0, np.flatnonzero(np.diff(xs) != 0) + 1, xs.size]
    jittered = xs.copy()
    for s, e in zip(b[:-1], b[1:]):
        L = e - s
        if L > 1:
            # centered ranks in [-0.5, 0.5]
            offs = (np.arange(L) - 0.5*(L-1)) / max(L-1, 1)
            jittered[s:e] += eps * offs
    x[order] = jittered
    return x

def detrend_palma_2pass_isotonic_stable(
    x, stat,
    mad_eps=1e-12,
    mad_topcut=0.01,   # keep r1 <= q_{1-mad_topcut}
    mad_nbins=401      # odd recommended
):
    x = _monotone_tie_jitter(x, eps=1e-9)
    x = np.asarray(x, float)
    y = np.log(np.maximum(np.asarray(stat, float), float(mad_eps)))

    # --- two-pass isotonic (decreasing) ---
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    yfit1 = iso.fit_transform(x, y)
    r1 = y - yfit1

    cutoff = np.quantile(r1, 1.0 - mad_topcut)
    keep = (r1 <= cutoff)
    assert keep.sum() >= 2, "Not enough inliers for second isotonic fit."

    iso2 = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso2.fit(x[keep], y[keep])
    yhat = iso2.predict(x)
    r = y - yhat

    # --- weighted median / MAD (same as yours) ---
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
            # IQR fallback, weighted
            o = np.argsort(v, kind='mergesort'); v, w = v[o], w[o]
            cw = np.cumsum(w); cw /= cw[-1]
            q25 = np.interp(0.25, cw, v); q75 = np.interp(0.75, cw, v)
            mad = max(0.7413 * (q75 - q25), eps)
        return mad

    # --- local kernel MAD over index (avoids h=0) ---
    def _local_kernel_mad_index(r, x, k):
        n = r.size
        k = int(max(3, min(k, n)))
        if k % 2 == 0:
            k -= 1
        half = k // 2

        order = np.argsort(x, kind="mergesort")
        rs = r[order]
        s_sorted = np.empty(n, float)

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            m  = hi - lo
            # triangular weights purely by index distance (center weight 1, edges 0)
            if m <= 1:
                s_sorted[i] = mad_eps
                continue
            idx = np.arange(lo, hi)
            w = 1.0 - np.abs(idx - i) / max(1, m - 1)
            s_sorted[i] = _wmad(rs[lo:hi], w, eps=mad_eps)

        s = np.empty_like(s_sorted)
        s[order] = s_sorted
        return s

    s = _local_kernel_mad_index(r, x, k=mad_nbins)
    z = r / s
    return r, z

from sklearn.isotonic import IsotonicRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

def detrend_palma_cont_scale(x, stat, *, span=0.15, topcut=0.01, eps=1e-12):
    x = np.asarray(x, float)
    y = np.log(np.maximum(np.asarray(stat, float), eps))
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    yfit1 = iso.fit_transform(x, y)
    r1 = y - yfit1
    keep = r1 <= np.quantile(r1, 1 - topcut)
    iso2 = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x[keep], y[keep])
    yhat = iso2.predict(x); r = y - yhat

    # smooth log|r| -> scale; calibrate to be comparable to MAD
    logabs = np.log(np.abs(r) + eps)
    log_s  = lowess(logabs, x, frac=span, it=0, return_sorted=False)
    s      = np.exp(log_s)

    # optional calibration so that median(s) matches 1.4826*median(|r|) globally
    target = 1.4826 * np.median(np.abs(r))
    s *= (target / np.median(s)) if np.median(s) > 0 else 1.0

    z = r / np.maximum(s, eps)
    return r, z