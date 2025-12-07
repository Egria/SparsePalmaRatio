# scripts/pca500_knn_leiden.py
# Run as: python scripts/pca500_knn_leiden.py

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.comparison import compare_clusters_filtered

os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ---- import your project entry points (do NOT modify repo code) ----
# They must already be importable from your repo; adapt the import path only if
# your project expects a specific module layout. The function names below are
# exactly what you used in your prompt.
from src.parameters import Parameters
from src.preprocess import preprocess
from src.filter import filter_counts  # if your function sits elsewhere, adjust only THIS import line


# ----------------- helpers that do not modify your repo code -----------------

def _align_labels(labels, cells_all: pd.Index, cells_used: pd.Index) -> np.ndarray:
    """
    Make a ground-truth vector aligned to 'cells_used' (filtered cells) without
    changing your original code. Accepts multiple 'labels' formats:
      - pd.Series indexed by cell IDs
      - pd.DataFrame with columns ['cell.ID','cluster.ID'] (common in your code)
      - 1D ndarray aligned to cells_all (the original order)
    """
    if isinstance(labels, pd.Series):
        s = labels.astype(str)
        if s.index.equals(cells_all):
            return s.reindex(cells_used).astype(str).to_numpy()
        # otherwise treat as mapping by ID
        return s.reindex(cells_used).astype(str).to_numpy()

    if isinstance(labels, pd.DataFrame):
        cols = [c.lower() for c in labels.columns]
        df = labels.copy()
        df.columns = cols
        if "cell.id" in df.columns and "cluster.id" in df.columns:
            s = df.set_index("cell.id")["cluster.id"].astype(str)
            return s.reindex(cells_used).astype(str).to_numpy()
        # fallback: first col = id, second col = label
        s = df.set_index(df.columns[0])[df.columns[1]].astype(str)
        return s.reindex(cells_used).astype(str).to_numpy()

    if isinstance(labels, np.ndarray):
        if labels.shape[0] != len(cells_all):
            raise ValueError("labels ndarray must be aligned to 'cells' returned by preprocess.")
        s = pd.Series(labels.astype(str), index=cells_all)
        return s.reindex(cells_used).astype(str).to_numpy()

    raise TypeError("Unsupported labels format; supply Series/DataFrame/ndarray.")


def _build_knn_graph_from_dense(Z: np.ndarray, k: int = 30, mode: str = "union") -> sp.csr_matrix:
    """
    Cosine kNN graph from dense features Z (cells × d).
    Returns a symmetric CSR similarity matrix with weights = 1 - cosine distance.
    """
    n = Z.shape[0]
    k = min(k, max(1, n - 1))
    nn = min(k + 1, n)

    nbrs = NearestNeighbors(n_neighbors=nn, metric="cosine", algorithm="brute", n_jobs=1)
    nbrs.fit(Z)
    dist, idx = nbrs.kneighbors(Z, return_distance=True)

    # drop self (first neighbor)
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    sim = 1.0 - dist

    rows = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    cols = idx.ravel()
    data = sim.ravel()
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    if mode == "union":
        A = A.maximum(A.T)
    elif mode == "mutual":
        A = A.minimum(A.T)
    else:
        raise ValueError("mode must be 'union' or 'mutual'")

    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def _leiden_from_similarity(A: sp.csr_matrix, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    """
    Leiden on a symmetric similarity graph A; returns labels (np.int32).
    Uses igraph + leidenalg; no changes to your repo code.
    """
    import igraph as ig
    import leidenalg as la

    # make undirected igraph with weights
    C = sp.triu(A, k=1, format="coo")
    edges = list(zip(C.row.tolist(), C.col.tolist()))
    g = ig.Graph(n=A.shape[0], edges=edges, directed=False)
    g.es["weight"] = C.data.tolist()

    try:
        la.set_rng_seed(int(seed))
    except Exception:
        pass

    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        n_iterations=-1
    )
    return np.array(part.membership, dtype=np.int32)


def _plot_tsne_precomputed_from_knn(
    A: sp.csr_matrix,            # symmetric similarity (cells × cells), e.g., union kNN with weights in [0,1]
    labels_true: np.ndarray,     # ground-truth labels aligned to A's rows
    out_png: str,
    title: str,
    *,
    fill_value: float = 1.0,     # distance for non-edges; keep in [0, 2] for cosine-derived sims
    perplexity: float = 30.0,
    random_state: int = 0
):
    """
    Build a dense precomputed distance matrix from the kNN graph and run t-SNE.

    D construction:
      - Start with D = fill_value (far distance) everywhere.
      - Set D[i,i] = 0.
      - For edges (i,j) in A: D[i,j] = 1 - sim_ij  (if A stores cosine similarity in [0,1]).
      - Symmetrize: D = max(D, D.T).

    WARNING: D is dense (n_cells × n_cells). This only fits for smaller datasets.
             If n is large, consider subsampling before calling this routine.
    """
    if not sp.isspmatrix_csr(A):
        A = A.tocsr(copy=False)
    n = A.shape[0]

    # --- dense distance matrix from sparse similarities ---
    D = np.full((n, n), float(fill_value), dtype=np.float32)
    np.fill_diagonal(D, 0.0)

    C = A.tocoo(copy=True)
    # Convert similarity -> distance; clamp to [0, fill_value]
    Dij = np.clip(1.0 - C.data.astype(np.float32, copy=False), 0.0, float(fill_value))
    D[C.row, C.col] = Dij

    # Ensure perfect symmetry (important for TSNE(metric='precomputed'))
    # Use the larger distance so non-edges remain 'far'
    D = np.maximum(D, D.T)

    # --- run t-SNE on the precomputed distances ---
    perpl = min(perplexity, max(5.0, (n - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        perplexity=perpl,
        init="random",
        learning_rate="auto",
        random_state=int(random_state),
        method="barnes_hut"  # default for 2D; fine to keep explicit
    )
    XY = tsne.fit_transform(D)

    # --- plot colored by ground truth labels ---
    labs = pd.Series(labels_true, dtype=str)
    uniq = np.array(sorted(pd.unique(labs)))
    lut = {u: plt.cm.tab20(i % 20) for i, u in enumerate(uniq)}
    colors = labs.map(lut).tolist()

    plt.figure(figsize=(8, 7), dpi=160)
    plt.scatter(XY[:, 0], XY[:, 1], c=colors, s=3, alpha=0.9, linewidths=0)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if len(uniq) <= 20:
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=lut[u], markersize=6, label=str(u))
                   for u in uniq]
        plt.legend(handles=handles, loc="best", fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ------------------------------- main pipeline -------------------------------

def main():
    # 0) config and I/O
    config_filename = "cfg/config_102580.json"
    params = Parameters(config_filename)

    # 1) read raw (repo function)
    # Expect: matrix (genes × cells, CSR), cells (Index), genes (Index), labels (Series/DF/ndarray)
    X, cells, genes, labels = preprocess(params)

    # 2) cell filtering (repo function) → we will use *all genes* downstream
    #    We only borrow the filtered cell set from here.
    X_f, genes_f, cells_f = filter_counts(params, X, genes, cells, save=False)

    # 3) Use ALL GENES + filtered cells
    #    X is genes×cells; keep all rows, only filtered columns, preserve order of cells_f
    col_idx = pd.Index(cells).get_indexer(pd.Index(cells_f))
    if np.any(col_idx < 0):
        raise ValueError("Some filtered cells not found in original 'cells' index.")
    X_allg = X[:, col_idx]  # genes × filtered_cells (CSR)
    cells_used = pd.Index(cells_f)
    genes_all = pd.Index(genes)  # all genes

    # 4) PCA (TruncatedSVD) to 500 comps on sparse; rows = cells
    n_pcs = 500
    n_cells = X_allg.shape[1]
    n_genes = X_allg.shape[0]
    n_pcs = int(min(n_pcs, max(2, min(n_cells - 1, n_genes))))  # safety

    svd = TruncatedSVD(n_components=n_pcs, random_state=getattr(params, "random_state", 0))
    Z = svd.fit_transform(X_allg.T.astype(np.float32, copy=False))  # cells × PCs
    Z = normalize(Z, norm="l2", axis=1, copy=False)                 # cosine-friendly

    # 5) kNN graph on PCs (cosine)
    k = int(getattr(params, "knn_k", 30))
    A = _build_knn_graph_from_dense(Z, k=k, mode="union")           # symmetric CSR (similarity)

    # 6) Leiden clustering on kNN graph
    res = float(getattr(params, "leiden_resolution", 1.0))
    seed = int(getattr(params, "random_state", 0))
    y_pred = _leiden_from_similarity(A, resolution=res, seed=seed)

    # 7) Align ground truth to filtered cells & score
    y_true = _align_labels(labels, pd.Index(cells), cells_used)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    print(f"[Leiden @ res={res:.2f}, k={k}]  ARI={ari:.4f}  NMI={nmi:.4f}")

    # 8) t‑SNE visualization (PC space used for kNN) colored by ground truth labels
    outdir = "result_pca500_102580"#getattr(params, "output_folder", "outputs")
    os.makedirs(outdir, exist_ok=True)
    _plot_tsne_precomputed_from_knn(
        A,  # your symmetric kNN similarity (CSR)
        y_true,  # ground truth aligned to A's rows
        os.path.join(outdir, "tsne_knn_precomputed.png"),
        title="t-SNE (precomputed distances from kNN graph)",
        fill_value=1.0,  # non-edges treated as far; adjust if your sim scale differs
        perplexity=30.0,
        random_state=seed
    )

    # 9) Save clustering labels for filtered cells
    pd.DataFrame({"cell.ID": cells_used, "cluster.ID": y_pred}).to_csv(
        os.path.join(outdir, "leiden_pc500_knn_labels.csv"), index=False
    )
    params.output_folder = outdir
    tab, gt_breakdown, ari, nmi = compare_clusters_filtered(params, y_pred, y_true, cells_f, cells)

    with open(os.path.join(outdir, "clustering_scores.txt"), "w") as f:
        f.write(f"ARI={ari:.6f}\nNMI={nmi:.6f}\n")



if __name__ == "__main__":
    main()