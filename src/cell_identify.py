import pandas as pd

import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from .parameters import Parameters
from .preprocess import preprocess
from .filter import filter_counts
from .calc_stat import calc_gene_stats
from .detrend import detrend
from .generate_cluster import generate_clusters
from .comparison import compare_clusters_filtered
from .channels import make_channel_graphs
from .refine_cluster import refine_cluster
from .cell_normalize import cpm_normalize_csr


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
    #uniq = ['Krt4/13+', 'Ciliated', 'Brush', 'Cycling Basal (homeostasis)',
    #        'Cycling Basal (regeneration)', 'Basal', 'Ionocytes', 'Pre-ciliated',
    #        'PNEC', 'Secretory']
    lut = {u: plt.cm.tab20(i % 20) for i, u in enumerate(uniq)}
    colors = labs.map(lut).tolist()

    plt.figure(figsize=(8, 7), dpi=160)
    plt.scatter(XY[:, 0], XY[:, 1], c=colors, s=3, alpha=0.9, linewidths=0)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    print(lut.keys())
    if len(uniq) <= 20:
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=lut[u], markersize=6, label=str(u))
                   for u in uniq]
        plt.legend(handles=handles, loc="best", fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

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

def cell_identify(config_filename:str):
    params = Parameters(config_filename)
    matrix, cells, genes, labels = preprocess(params)
    matrix_f, genes_f, cells_f = filter_counts(params, matrix, genes, cells, False)
    #matrix_f = cpm_normalize_csr(matrix_f)
    #print(matrix_f)
    #return
    gene_stats = calc_gene_stats(params, matrix_f, genes_f)
    gene_stats = detrend(params, gene_stats)

    b5 = {"fano": 1.0}
    b4 = {"gini": 1.0}
    b3 = {"palma": 1.0}
    b2 = {"gini": 0.45, "palma": 0.55}
    b1 = {"gini": 0.6, "palma": 0.4}

    bands = [
        ("50-30", b5, 0.0, 1000),
        ("30-10", b4, 0.0, 1000),
        ("10-3", b3, 3.5, 1000),
        ("3-1", b2, 0.0, 1000),
        ("1-0.1", b1, 3.5, 1000)
    ]

    y_true = _align_labels(labels, pd.Index(cells), cells_f)

    GF, _ = make_channel_graphs(params, gene_stats, matrix_f, genes_f, labels, cells_f, cells, bands=bands,
                                            band_weights=[1.0, 0.0, 0.0, 0.0, 0.0])
    labels_f = generate_clusters(params, GF, cells_f)
    #y_pred = _align_labels(labels, pd.Index(cells), cells_f)

    _plot_tsne_precomputed_from_knn(
        GF,  # your symmetric kNN similarity (CSR)
        y_true,  # ground truth aligned to A's rows
        os.path.join(params.output_folder, "tsne_gfano.png"),
        title="t-SNE (precomputed distances from kNN graph)",
        fill_value=1.0,  # non-edges treated as far; adjust if your sim scale differs
        perplexity=30.0,
        random_state=0
    )
    #_plot_tsne_precomputed_from_knn(
    #    GF,  # your symmetric kNN similarity (CSR)
    #    y_pred,  # ground truth aligned to A's rows
    #    os.path.join(params.output_folder, "tsne_gfano_pred.png"),
    #    title="t-SNE (precomputed distances from kNN graph)",
    #    fill_value=1.0,  # non-edges treated as far; adjust if your sim scale differs
    #    perplexity=30.0,
    #    random_state=0
    #)

    GG, _ = make_channel_graphs(params, gene_stats, matrix_f, genes_f, labels, cells_f, cells, bands=bands,
                                band_weights=[0.0, 1.0, 0.0, 0.0, 0.0])
    labels_f = generate_clusters(params, GG, cells_f)
    y_pred = _align_labels(labels, pd.Index(cells), cells_f)
    _plot_tsne_precomputed_from_knn(
        GG,  # your symmetric kNN similarity (CSR)
        y_true,  # ground truth aligned to A's rows
        os.path.join(params.output_folder, "tsne_ggini.png"),
        title="t-SNE (precomputed distances from kNN graph)",
        fill_value=1.0,  # non-edges treated as far; adjust if your sim scale differs
        perplexity=30.0,
        random_state=0
    )
    
    #_plot_tsne_precomputed_from_knn(
    #    GG,  # your symmetric kNN similarity (CSR)
    #    y_pred,  # ground truth aligned to A's rows
    #    os.path.join(params.output_folder, "tsne_ggini_pred.png"),
    #    title="t-SNE (precomputed distances from kNN graph)",
    #    fill_value=1.0,  # non-edges treated as far; adjust if your sim scale differs
    #    perplexity=30.0,
    #    random_state=0
    #)

    GP, _ = make_channel_graphs(params, gene_stats, matrix_f, genes_f, labels, cells_f, cells, bands=bands,
                                band_weights=[0.0, 0.0, 1.0, 0.0, 0.0])
    labels_f = generate_clusters(params, GP, cells_f)
    y_pred = _align_labels(labels, pd.Index(cells), cells_f)
    _plot_tsne_precomputed_from_knn(
        GP,  # your symmetric kNN similarity (CSR)
        y_true,  # ground truth aligned to A's rows
        os.path.join(params.output_folder, "tsne_gpalma.png"),
        title="t-SNE (precomputed distances from kNN graph)",
        fill_value=1.0,  # non-edges treated as far; adjust if your sim scale differs
        perplexity=30.0,
        random_state=0
    )

    #_plot_tsne_precomputed_from_knn(
    #    GP,  # your symmetric kNN similarity (CSR)
    #    y_pred,  # ground truth aligned to A's rows
    #    os.path.join(params.output_folder, "tsne_gpalma_pred.png"),
    #    title="t-SNE (precomputed distances from kNN graph)",
    #    fill_value=1.0,  # non-edges treated as far; adjust if your sim scale differs
    #    perplexity=30.0,
    #    random_state=0
    #)


    graph, band_genes = make_channel_graphs(params, gene_stats, matrix_f, genes_f, labels, cells_f, cells, bands=bands,band_weights=[0.4, 0.4, 0.4, 0.0, 0.0])

    _plot_tsne_precomputed_from_knn(
        graph,  # your symmetric kNN similarity (CSR)
        y_true,  # ground truth aligned to A's rows
        os.path.join(params.output_folder, "tsne_gmix.png"),
        title="t-SNE (precomputed distances from kNN graph)",
        fill_value=1.0,  # non-edges treated as far; adjust if your sim scale differs
        perplexity=30.0,
        random_state=0
    )






    labels_f = generate_clusters(params, graph, cells_f)

    #from .detrending.gini_lowess import lowess_twopass_detrending
    #from .refine_cluster import refine_cluster_new
    #labels_rf = refine_cluster_new(
    #    params=params,
    #    mat=matrix_f,
    #    genes_index=genes_f,  # pd.Index of gene names
    #    labels=labels_f,
    #    global_graph=graph,
    #    lowess_fun=lowess_twopass_detrending
    #)

    labels_rf = refine_cluster(params, matrix_f, genes_f, labels_f, band_genes, graph)

    #y_pred = _align_labels(labels, pd.Index(cells), cells_f)

    #_plot_tsne_precomputed_from_knn(
    #    graph,  # your symmetric kNN similarity (CSR)
    #    y_pred,  # ground truth aligned to A's rows
    #    os.path.join(params.output_folder, "tsne_gmix_pred.png"),
    #    title="t-SNE (precomputed distances from kNN graph)",
    #    fill_value=1.0,  # non-edges treated as far; adjust if your sim scale differs
    #    perplexity=30.0,
    #    random_state=0
    #)





    result = pd.DataFrame({"label":labels_rf}, index=cells_f)
    tab, gt_breakdown, ari, nmi = compare_clusters_filtered(params, labels_rf, labels, cells_f, cells)
    #print(result["label"].value_counts())



