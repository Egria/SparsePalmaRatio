import numpy as np
import pandas as pd
from .parameters import Parameters
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def compare_clusters_filtered(
    params: Parameters,
    labels_f: np.ndarray,           # predicted cluster ids for filtered cells (e.g., ints, may include -1)
    labels_all: np.ndarray,         # ground-truth labels for ALL cells (same order as `cells`)
    cells_f: pd.Index,              # filtered cell IDs (subset of `cells`)
    cells: pd.Index,                # ALL cell IDs (same order as `labels_all`)
) -> tuple[pd.DataFrame, float, float]:
    """
    Compare predicted cluster labels vs. ground truth on *filtered* cells only.

    Returns
    -------
    cluster_analysis_table : pd.DataFrame (index = predicted cluster id; includes row -1)
        Columns: ['count','mapped_label','frequency','hit','precision','recall','entropy','f1']
        All statistics computed over the filtered subset only.
    ari : float
    nmi : float
    """
    # ---------------------------
    # 0) Map filtered cell IDs to ground-truth labels (subset only)
    # ---------------------------
    if not isinstance(cells, pd.Index):
        cells = pd.Index(cells)
    if not isinstance(cells_f, pd.Index):
        cells_f = pd.Index(cells_f)

    # ensure every filtered cell exists in `cells`
    pos = cells.get_indexer(cells_f)
    if np.any(pos < 0):
        missing = cells_f[np.flatnonzero(pos < 0)[:5]]
        raise ValueError(f"Some filtered cells not found in `cells`, e.g. {list(missing)} ...")

    y_true_f = labels_all[pos]                 # ground truth for filtered cells
    y_pred_f = np.asarray(labels_f)            # predictions for filtered cells
    n_f = y_pred_f.shape[0]
    if y_true_f.shape[0] != n_f:
        raise ValueError("Length mismatch between filtered predictions and mapped ground truth.")

    # ---------------------------
    # 1) Per-predicted-cluster table scaffold (filtered cells only)
    # ---------------------------
    # counts per predicted label
    pred, counts = np.unique(y_pred_f, return_counts=True)
    # Build table indexed by predicted cluster id; allow mixing in -1 even if absent
    idx = pd.Index(pred, dtype=object, name="cluster.ID")
    tab = pd.DataFrame(index=idx)
    tab["count"] = counts
    tab["frequency"] = tab["count"] / float(n_f)
    tab["mapped_label"] = "N/A"
    tab["hit"] = 0
    tab["precision"] = 0.0
    tab["recall"] = 0.0
    tab["entropy"] = 0.0
    tab["f1"] = 0.0

    # Precompute ground-truth counts (for recall denominators) on filtered set
    gt_counts = pd.Series(y_true_f, dtype="object").value_counts()

    # ---------------------------
    # 2) Majority-vote mapping & per-cluster stats
    # ---------------------------
    # Work cluster-by-cluster on filtered arrays
    y_true_series = pd.Series(y_true_f, index=cells_f, dtype="object")
    y_pred_series = pd.Series(y_pred_f, index=cells_f)

    for c in tab.index:
        # members of predicted cluster c (filtered only)
        members_mask = (y_pred_series.values == c)
        m = int(members_mask.sum())
        if m == 0:
            # leave zeros/defaults (keeps -1 row even if empty)
            continue

        # ground-truth label distribution within this cluster
        gt_sub = y_true_series.values[members_mask]
        pool = pd.Series(gt_sub, dtype="object").value_counts()

        # Shannon entropy (base-2) of the label distribution
        p = (pool / pool.sum()).to_numpy(dtype=float)
        entropy = float(-(p * np.log2(p)).sum()) if p.size else 0.0

        # winner by majority vote
        winner = pool.idxmax()
        hit = int(pool.loc[winner])
        precision = hit / m
        # recall denominator = total #filtered cells of that GT label
        denom = int(gt_counts.get(winner, 0))
        recall = (hit / denom) if denom > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # write stats
        tab.loc[c, "mapped_label"] = str(winner)
        tab.loc[c, "hit"] = hit
        tab.loc[c, "precision"] = precision
        tab.loc[c, "recall"] = recall
        tab.loc[c, "entropy"] = entropy
        tab.loc[c, "f1"] = f1

    # sort rows by cluster id (put -1 first, then ascending others)
    def _sort_key(val):
        try:
            v = int(val)
            return (v != -1, v)
        except Exception:
            return (1, str(val))
    tab = tab.sort_index(key=lambda s: s.map(_sort_key))

    # ---------------------------
    # 3) Overall metrics (filtered cells only; include -1 as a regular label)
    # ---------------------------
    # ARI/NMI accept arbitrary label types; pass as given
    ari = adjusted_rand_score(y_true_f, y_pred_f)
    nmi = normalized_mutual_info_score(y_true_f, y_pred_f, average_method="arithmetic")

    # ---------------------------
    # 4) Persist (optional)
    # ---------------------------

    tab.to_csv(f"{params.output_folder}/cluster_analysis.csv", index=True, header=True)
    with open(f"{params.output_folder}/overall_performance.txt", "w") as f:
        f.write(f"ARI: {ari:0.4f}, NMI: {nmi:0.4f}\n")

    return tab, ari, nmi
