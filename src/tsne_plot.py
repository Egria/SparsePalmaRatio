import os

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from .parameters import Parameters
from threadpoolctl import threadpool_limits, threadpool_info
from matplotlib.colors import hsv_to_rgb, to_hex

def make_freq_ranked_colors(labels, *,
                            base_hue=0.0,     # 0.0 = red; change to 0.03 for orange, etc.
                            s_max=0.98, s_min=0.25,
                            v_max=0.98, v_min=0.80):
    """
    Map each label to a distinct color where rarer labels are more vivid.
    Colors are generated in rarity order using evenly-spaced hues (golden-ratio),
    with saturation/value decreasing toward common labels.

    Returns
    -------
    color_by_label : dict {label -> hex color}
    order_common_first : pd.Index of labels sorted by frequency (common → rare)
    order_rare_first   : pd.Index of labels sorted by frequency (rare → common)
    """
    lab_series = pd.Series(labels, dtype="object")
    counts = lab_series.value_counts()
    order_rare_first   = counts.sort_values(ascending=True).index
    order_common_first = counts.sort_values(ascending=False).index

    L = len(order_rare_first)
    if L == 0:
        return {}, order_common_first, order_rare_first

    # Golden-ratio hue step gives well-separated hues
    phi = 0.61803398875

    color_by_label = {}
    for i, lab in enumerate(order_rare_first):
        t = i / max(1, L - 1)             # 0 for rarest … 1 for most common
        h = (base_hue + i * phi) % 1.0    # distinct hue per rank (rarest starts at red)
        s = s_max - (s_max - s_min) * t   # saturation fades with frequency
        v = v_max - (v_max - v_min) * t   # value slightly fades too
        rgb = hsv_to_rgb((h, s, v))
        color_by_label[lab] = to_hex(rgb)

    return color_by_label, order_common_first, order_rare_first

def tsne_from_feature(
    params:Parameters,
    B: sp.csr_matrix,
    labels_all: np.ndarray,  # ground-truth labels for ALL cells (same order as `cells`)
    cells_f: pd.Index,  # filtered cell IDs (subset of `cells`)
    cells: pd.Index,  # ALL cell IDs (same order as `labels_all`)
    perplexity: float = 30.0,
    random_state: int = 42,
    dtype = np.float32,
    l2_normalize: bool = True,
    output_suffix: str = ''
):
    """
    Build a 2D t-SNE layout from a sparse symmetric graph of filtered cells,
    and color by ground-truth labels.

    Returns
    -------
    emb_df : pd.DataFrame with columns ['tSNE1','tSNE2','label']
    fig, ax : matplotlib Figure and Axes
    """
    # 0) sanitize inputs

    pos = cells.get_indexer(cells_f)
    if np.any(pos < 0):
        missing = cells_f[np.flatnonzero(pos < 0)[:5]]
        raise ValueError(f"Some filtered cells not found in `cells`, e.g. {list(missing)} ...")

    labels_f = labels_all[pos].astype(object)  # ground truth for filtered cells

    if not sp.isspmatrix_csr(B):
        B = B.tocsr(copy=False)
    B.eliminate_zeros()
    n_cells, n_features = B.shape
    y = np.asarray(labels_f)
    if y.shape[0] != n_cells:
        raise ValueError("labels_f length must equal #rows of B")

    # 1) optional row L2 normalization (keeps sparsity)
    X = normalize(B, norm="l2", axis=1, copy=True) if l2_normalize else B

    # 2) optional SVD → dense (n_cells × n_svd)


    # 3) t-SNE on the small dense table
    with threadpool_limits(limits=8, user_api="blas"):
        X_small = X.toarray().astype(dtype, copy=False)

        # 3) t-SNE (avoid PCA init → no extra SVD)
        perpl = min(perplexity, max(5.0, (n_cells - 1) / 3.0))
        tsne = TSNE(
            n_components=2,
            perplexity=perpl,
            init="random",  # <— avoid PCA → no extra BLAS call
            learning_rate="auto",
            random_state=random_state,
            metric="euclidean",
            method="barnes_hut"  # default for n_components=2; set explicit
        )
        Y = tsne.fit_transform(X_small)

    # 4) assemble and plot
    emb_df = pd.DataFrame({"tSNE1": Y[:, 0], "tSNE2": Y[:, 1], "label": pd.Series(y, dtype="object")})
    labels = emb_df["label"].astype("object")
    color_by_label, common_first, rare_first = make_freq_ranked_colors(labels)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw common first so rare (vivid) points overplot them
    for lab in common_first:
        m = (labels.to_numpy() == lab)
        ax.scatter(
            emb_df.loc[m, "tSNE1"], emb_df.loc[m, "tSNE2"],
            s=6.0, linewidths=0,
            color=color_by_label[lab],
            alpha=0.9 if lab in rare_first[:max(1, len(rare_first) // 6)] else 0.75,
            label=f"{lab} (n={m.sum()})",
            zorder=1
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE — rare labels use vivid, distinct colors")
    ax.legend(loc="best", frameon=True, fontsize=8, markerscale=2, handletextpad=0.3, borderpad=0.3)
    fig.tight_layout()
    plt.savefig(f"{params.output_folder}/tsne_{output_suffix}.png", dpi=200)
    return emb_df