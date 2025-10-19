import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from typing import Tuple

# ---------- regex & helpers ----------

_B1_RE = re.compile(r"^(?P<parent>.+)_B1_(?P<id>\d+)$")

def _majors_from_labels(labels_all: pd.Series | np.ndarray) -> np.ndarray:
    labs = pd.Series(labels_all, dtype="object", copy=False)
    majors = []
    for s in labs.astype(str):
        m = _B1_RE.match(s)
        majors.append(m.group("parent") if m else s)
    return np.asarray(majors, dtype=object)

def _indexer_first(gf: pd.Index, gq) -> np.ndarray:
    gq = pd.Index(gq)
    if gf.is_unique:
        idx = gf.get_indexer(gq)
        return idx[idx >= 0].astype(np.int64)
    pos = pd.Series(np.arange(len(gf), dtype=np.int64), index=gf)
    first = pos.groupby(level=0).nth(0)
    idx_s = first.reindex(gq)
    return idx_s[idx_s.notna()].astype(np.int64).to_numpy()

def _arctan_center_rows(X_rows: sp.csr_matrix) -> sp.csr_matrix:
    if not sp.isspmatrix_csr(X_rows):
        X_rows = X_rows.tocsr(copy=False)
    X_rows.sort_indices()
    n_feat, n_cells = X_rows.shape
    indptr, data = X_rows.indptr, X_rows.data
    for r in range(n_feat):
        s, e = indptr[r], indptr[r + 1]
        if s == e: continue
        v = data[s:e].astype(np.float64, copy=False)
        vs = np.sort(v)[::-1]
        S = float(vs.sum());
        if S <= 0.0: continue
        th = 0.8 * S
        binCellNum = int(n_cells // 1000)
        if binCellNum <= 9:
            cs = np.cumsum(vs); j_end = int(np.searchsorted(cs, th, side="right")) + 1
        else:
            loopNum = int((n_cells - binCellNum) // binCellNum)
            if loopNum <= 0:
                cs = np.cumsum(vs); j_end = int(np.searchsorted(cs, th, side="right")) + 1
            else:
                ends = (np.arange(loopNum, dtype=np.int64) + 1) * binCellNum
                cs = np.cumsum(vs)
                clamp = np.minimum(ends, vs.size)
                sums_at_end = cs[clamp - 1]
                hit = sums_at_end > th
                j_end = int(ends[int(np.flatnonzero(hit)[0])]) if np.any(hit) else int(ends[-1])
        expM = float((np.cumsum(vs)[j_end - 1] / j_end) if j_end <= vs.size else (S / j_end))
        data[s:e] = 10.0 * (np.arctan(v - expM) + np.arctan(expM))
    X_rows.eliminate_zeros()
    return X_rows

def _zscore_rows_global(X_rows: sp.csr_matrix) -> sp.csr_matrix:
    if not sp.isspmatrix_csr(X_rows):
        X_rows = X_rows.tocsr(copy=False)
    X_rows = X_rows.astype(np.float32, copy=False)
    n_cells = X_rows.shape[1]
    sums = np.asarray(X_rows.sum(axis=1)).ravel()
    mu = sums / float(n_cells)
    Xsq = X_rows.copy(); Xsq.data **= 2
    sums2 = np.asarray(Xsq.sum(axis=1)).ravel()
    var = (sums2 / float(n_cells)) - (mu ** 2)
    var[var < 0] = 0
    std = np.sqrt(var, dtype=np.float32)
    rows = np.repeat(np.arange(X_rows.shape[0], dtype=np.int64), np.diff(X_rows.indptr))
    X_rows.data -= mu[rows]
    nz = std[rows] > 0
    X_rows.data[nz] /= std[rows[nz]]
    X_rows.eliminate_zeros()
    return X_rows

def _idf_weights_from_rows(X_rows: sp.csr_matrix, smooth: bool = True) -> np.ndarray:
    # df = number of non-zeros per row (gene)
    df = np.diff(X_rows.indptr).astype(np.float64)
    n = X_rows.shape[1]
    if smooth:
        w = np.log((1.0 + n) / (1.0 + df)) + 1.0
    else:
        w = np.log(np.maximum(n / np.maximum(df, 1.0), 1.0))
    return w.astype(np.float32)

def _l2_normalize_rows_csr(X: sp.csr_matrix) -> sp.csr_matrix:
    if not sp.isspmatrix_csr(X):
        X = X.tocsr(copy=False)
    X = X.astype(np.float32, copy=False)
    sq = X.copy(); sq.data **= 2
    norms = np.sqrt(np.asarray(sq.sum(axis=1)).ravel(), dtype=np.float64)
    norms = np.maximum(norms, 1e-12)
    rows = np.repeat(np.arange(X.shape[0], dtype=np.int64), np.diff(X.indptr))
    X.data = (X.data / norms[rows]).astype(np.float32, copy=False)
    return X

def _knn_graph_cosine_deterministic_from_csr(C: sp.csr_matrix, k: int = 20, mode: str = "union") -> sp.csr_matrix:
    # cosine = 1 - dot for L2-normalized rows in a brute-force neighbor search
    n = C.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n), metric="cosine", algorithm="brute", n_jobs=1)
    nbrs.fit(C)
    dist, idx = nbrs.kneighbors(C, return_distance=True)  # sorted
    dist = dist[:, 1:]; idx = idx[:, 1:]
    sim = (1.0 - dist).astype(np.float32, copy=False)
    rows = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    cols = idx.ravel(); dat = sim.ravel()
    A = sp.csr_matrix((dat, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T) if mode == "union" else A.minimum(A.T)
    A.setdiag(0.0); A.eliminate_zeros()
    # row-stochastic P
    rs = np.asarray(A.sum(axis=1)).ravel()
    Dinv = sp.diags(1.0 / np.maximum(rs, 1e-12))
    P = Dinv @ A
    return P

def _marker_auc_logfc(X_rows: sp.csr_matrix, gene_idx: np.ndarray, S_mask: np.ndarray, eps: float = 1e-9):
    Xg = X_rows[gene_idx, :]
    y = S_mask.astype(np.uint8)
    sel = np.where(S_mask)[0]; rest = np.where(~S_mask)[0]
    if sel.size == 0 or rest.size == 0:
        return np.zeros(len(gene_idx)), np.zeros(len(gene_idx))
    mean_pos = np.asarray(Xg[:, sel].mean(axis=1)).ravel()
    mean_neg = np.asarray(Xg[:, rest].mean(axis=1)).ravel()
    lfc = np.log2((mean_pos + eps) / (mean_neg + eps))
    auc = np.zeros(len(gene_idx), dtype=float)
    for i in range(len(gene_idx)):
        xi = np.asarray(Xg[i, :].todense()).ravel() if sp.issparse(Xg) else Xg[i, :]
        try:
            auc[i] = roc_auc_score(y, xi)
        except Exception:
            auc[i] = 0.5
    return auc, lfc

def _centroid_cosine_scores(C: sp.csr_matrix, S_idx: np.ndarray) -> np.ndarray:
    """Cosine sim of every row in C to the L2-normalized centroid of rows S_idx."""
    if S_idx.size == 0:
        return np.zeros(C.shape[0], dtype=np.float64)
    c = np.asarray(C[S_idx, :].mean(axis=0)).ravel()
    den = np.linalg.norm(c)
    if den == 0.0 or not np.isfinite(den):
        return np.zeros(C.shape[0], dtype=np.float64)
    c = (c / den).astype(np.float64, copy=False)
    sim = C.dot(c)
    return np.asarray(sim).ravel()

# ---------- main: cell-wise B1 augmentation (no PCA) ----------

def augment_B1_children_from_labels(
    X_gc: sp.csr_matrix,
    genes_index: pd.Index,
    labels_all: np.ndarray | pd.Series,     # majors + '<parent>_B1_<id>'
    B1_genes,
    *,
    use_arctan: bool = True,
    use_idf: bool = False,
    k_knn: int = 20,
    quantile_gate: float = 0.10,            # q10 gate per metric (sim/signature/prop)
    min_seed_neighbors: int = 2,            # require ≥ this many kNN neighbors in the seed set
    allow_steal: bool = False,              # if False, do not reassign cells already in some B1 child
    auc_min: float = 0.85,
    n_markers_min: int = 1,
    max_rounds: int = 1,                    # rounds of re-estimating centroid/markers after additions
    single_thread: bool = True,
    output_path: str = '.'
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    PCA-free, per-cell augmentation of B1 children across other majors.
    Each candidate cell is judged independently; accepted cells are added individually.

    Returns
    -------
    labels_aug : np.ndarray (object)
    report     : pd.DataFrame with one row per ACCEPTED cell (child, cell, scores, thresholds).
    """
    # determinism
    if single_thread:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Basic shapes & labels
    if not sp.isspmatrix_csr(X_gc):
        X_gc = X_gc.tocsr(copy=False)
    X_gc.eliminate_zeros()
    n_genes, n_cells = X_gc.shape

    labels_all = pd.Series(labels_all, dtype=object).reset_index(drop=True)
    majors = _majors_from_labels(labels_all)

    # Identify B1 children present
    uniq = pd.Index(labels_all.unique())
    child_labels = [lab for lab in uniq if _B1_RE.match(str(lab))]
    if len(child_labels) == 0:
        return labels_all.to_numpy(dtype=object), pd.DataFrame([], columns=["child","cell","sim","sig","prop","comp","accepted"])

    # Build global B1 matrix (genes × cells)
    idx_B1 = _indexer_first(genes_index, B1_genes)
    if idx_B1.size == 0:
        raise ValueError("No B1 genes found in genes_index.")
    X_b1 = X_gc[idx_B1, :].tocsr(copy=False)
    if use_arctan:
        X_b1 = _arctan_center_rows(X_b1)
    X_b1 = _zscore_rows_global(X_b1)
    if use_idf:
        w_idf = _idf_weights_from_rows(X_b1, smooth=True)
        X_b1 = sp.diags(w_idf) @ X_b1

    # cells × genes CSR, L2 rows
    C = X_b1.T.tocsr(copy=False).astype(np.float32, copy=False)
    C = _l2_normalize_rows_csr(C)

    # deterministic kNN → row-stochastic P
    P = _knn_graph_cosine_deterministic_from_csr(C, k=k_knn, mode="union")  # (n_cells × n_cells)

    labels_out = labels_all.copy()
    report_rows = []

    for round_idx in range(int(max_rounds)):
        # Collect per-child candidate tables
        per_child_tables = []  # list of DataFrames with columns: cell, child, sim, sig, prop, comp

        for child in child_labels:
            S = np.flatnonzero(labels_out.values == child)
            if S.size < 3:
                continue
            parent = _B1_RE.match(str(child)).group("parent")
            outside_parent = (majors != parent)

            # If disallow stealing, also forbid cells already belonging to any B1 child
            if not allow_steal:
                is_B1 = labels_out.astype(str).str.contains(r"_B1_\d+$", regex=True).to_numpy()
                eligible_pool = outside_parent & (~is_B1)
            else:
                eligible_pool = outside_parent

            # 1) centroid cosine similarity in B1 space
            sim_pc = _centroid_cosine_scores(C, S)  # (n_cells,)

            # 2) signature score using positive B1 markers (global)
            S_mask = np.zeros(n_cells, dtype=bool); S_mask[S] = True
            auc, lfc = _marker_auc_logfc(X_b1, np.arange(X_b1.shape[0]), S_mask)
            pos = (auc >= auc_min) & (lfc > 0)
            if pos.sum() < n_markers_min:
                # cannot generalize robustly from this child
                continue
            w = np.clip((auc - auc_min) / max(1e-6, 1.0 - auc_min), 0.0, 1.0)
            w[~pos] = 0.0
            s_sig = X_b1.T.dot(w).astype(np.float64, copy=False)  # (n_cells,)

            # 3) one-step neighbor support (row-stochastic P)
            y0 = np.zeros(n_cells, dtype=np.float64); y0[S] = 1.0 / S.size
            prop = P @ y0  # (n_cells,)

            # 4) seed-derived gates (q10 of seed distributions)
            q = float(quantile_gate)
            thr_sim = float(np.quantile(sim_pc[S], q))
            thr_sig = float(np.quantile(s_sig[S], q))
            thr_prop = float(np.quantile(prop[S], q))

            # 5) local seed-neighbor count (must have >= min_seed_neighbors)
            #    counts = (binary adjacency) · 1_S
            A01 = P.copy(); A01.data[:] = 1.0
            seed_vec = np.zeros(n_cells, dtype=np.float32)
            seed_vec[S] = 1.0
            n_seed_nbr = (A01 @ seed_vec).astype(np.float32)  # (n_cells,)

            # Candidates that pass all per-cell gates
            cand_mask = eligible_pool & (~S_mask) \
                        & (sim_pc >= thr_sim) & (s_sig >= thr_sig) & (prop >= thr_prop) \
                        & (n_seed_nbr >= float(min_seed_neighbors))
            cand_idx = np.flatnonzero(cand_mask)
            if cand_idx.size == 0:
                continue

            # Composite score: normalize each metric to [0,1] wrt seed range (q10→q90), then average
            def _norm01(x_all, seeds, thr):
                hi = float(np.quantile(x_all[seeds], 0.90))
                num = np.clip(x_all - thr, 0.0, None); den = max(hi - thr, 1e-8)
                return np.clip(num / den, 0.0, 1.0)

            sim_n = _norm01(sim_pc, S, thr_sim)
            sig_n = _norm01(s_sig,  S, thr_sig)
            prop_n = _norm01(prop,  S, thr_prop)
            comp = (sim_n + sig_n + prop_n) / 3.0

            per_child_tables.append(pd.DataFrame({
                "cell": cand_idx,
                "child": child,
                "sim": sim_pc[cand_idx],
                "sig": s_sig[cand_idx],
                "prop": prop[cand_idx],
                "comp": comp[cand_idx],
                "thr_sim": thr_sim,
                "thr_sig": thr_sig,
                "thr_prop": thr_prop,
                "round": round_idx + 1
            }))

        if not per_child_tables:
            break

        # Resolve cross-child conflicts deterministically: highest comp, then sim, then smallest cell id, then child name
        claims = pd.concat(per_child_tables, ignore_index=True)
        claims.sort_values(by=["cell", "comp", "sim", "child"],
                           ascending=[True, False, False, True],
                           kind="mergesort", inplace=True)
        best = claims.drop_duplicates(subset=["cell"], keep="first")

        # Accept cells individually
        for _, row in best.iterrows():
            cell = int(row["cell"]); child = row["child"]
            labels_out.iloc[cell] = child
            report_rows.append({
                "round": int(row["round"]),
                "child": child,
                "cell": cell,
                "sim": float(row["sim"]),
                "sig": float(row["sig"]),
                "prop": float(row["prop"]),
                "comp": float(row["comp"]),
                "thr_sim": float(row["thr_sim"]),
                "thr_sig": float(row["thr_sig"]),
                "thr_prop": float(row["thr_prop"]),
                "accepted": True
            })

        # loop again if max_rounds > 1 (centroids/markers update with new members)
        if round_idx + 1 >= int(max_rounds):
            break

    report = pd.DataFrame(report_rows)
    report.to_csv(f"{output_path}/harvest_b1.csv", index=True, header=True)
    return labels_out.to_numpy(dtype=object), report
