import numpy as np
import scipy.sparse as sp

def knn_graph_from_binary_B(
    B_features_x_cells: sp.csr_matrix,
    k: int = 30,
    mode: str = "union",           # 'union' (kNN ∪ kNN^T) or 'mutual' (kNN ∩ kNN^T)
    sym_weight: str = "max",       # how to combine weights on symmetrization: 'max' | 'mean' | 'min'
    block_size: int = 4096,        # cells processed per block (tune for memory)
    ensure_binary: bool = True,    # treat any nonzero as 1
    return_distance: bool = False  # False -> similarity (for Leiden); True -> 1 - similarity
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Build a sparse kNN graph from a binary feature matrix B (features x cells), using Jaccard.
    Returns:
      G : csr_matrix (n_cells x n_cells), symmetric, weights = similarity (or distance if return_distance=True)
      zero_cells : boolean mask (n_cells,), True where a cell has no active features
    """
    # --- 0) Validate & orient ---
    if not sp.isspmatrix_csr(B_features_x_cells):
        B_features_x_cells = B_features_x_cells.tocsr(copy=False)
    B_features_x_cells.eliminate_zeros()

    # Ensure binary (no densify)
    if ensure_binary and B_features_x_cells.data.size:
        B_features_x_cells.data[:] = 1

    # cells x features for row-wise (cell-wise) operations
    B = B_features_x_cells.T.tocsr(copy=False)   # shape: (n_cells x n_features)
    B.sort_indices()

    n_cells = B.shape[0]
    # per-cell set sizes (# of 1s)
    r = np.diff(B.indptr).astype(np.int64)
    zero_cells = (r == 0)

    # Precompute transpose once for intersections
    BT = B.T.tocsc(copy=False)

    # Storage for directed kNN edges
    rows_all, cols_all, w_all = [], [], []

    # --- 1) Blocked intersections and per-row top-k selection ---
    for i0 in range(0, n_cells, block_size):
        i1 = min(n_cells, i0 + block_size)
        Bi = B[i0:i1, :]                       # (b x f)
        # Sparse intersections: (b x n_cells)
        C = Bi.dot(BT).tocsr(copy=False)
        C.eliminate_zeros()
        indptr, indices, data = C.indptr, C.indices, C.data
        r_block = r[i0:i1]

        for off in range(i1 - i0):
            i = i0 + off
            s, e = indptr[off], indptr[off + 1]
            if s == e:
                continue  # no overlap with anyone
            js = indices[s:e]
            inter = data[s:e].astype(np.int64, copy=False)

            # exclude self
            self_mask = (js != i)
            if not np.any(self_mask):
                continue
            js = js[self_mask]
            inter = inter[self_mask]

            # Jaccard similarity: inter / (|A| + |B| - inter)
            denom = r_block[off] + r[js] - inter
            # safe guard: denom >= 1 when inter>0; keep only valid
            valid = denom > 0
            if not np.any(valid):
                continue
            js = js[valid]
            inter = inter[valid]
            denom = denom[valid]
            sim = inter / denom.astype(np.float64, copy=False)

            # select top-k by similarity
            k_eff = min(k, sim.size)
            top_idx = np.argpartition(sim, -k_eff)[-k_eff:]
            # rank them (descending similarity)
            order = top_idx[np.argsort(sim[top_idx])[::-1]]

            rows_all.append(np.full(order.size, i, dtype=np.int64))
            cols_all.append(js[order])
            w_all.append(sim[order].astype(np.float64, copy=False))

    if rows_all:
        rows_dir = np.concatenate(rows_all)
        cols_dir = np.concatenate(cols_all)
        w_dir = np.concatenate(w_all)
    else:
        rows_dir = np.array([], dtype=np.int64)
        cols_dir = np.array([], dtype=np.int64)
        w_dir = np.array([], dtype=np.float64)

    # Directed kNN adjacency
    A = sp.csr_matrix((w_dir, (rows_dir, cols_dir)), shape=(n_cells, n_cells))
    A.eliminate_zeros()

    # --- 2) Symmetrize to undirected (kNN union or mutual) ---
    AT = A.T.tocsr(copy=False)

    if mode == "union":
        if sym_weight == "max":
            W = A.maximum(AT)
        elif sym_weight == "mean":
            W = (A + AT) * 0.5
        elif sym_weight == "min":
            W = A.minimum(AT)
        else:
            raise ValueError("sym_weight must be 'max', 'mean', or 'min'")

    elif mode == "mutual":
        # keep edges that appear in both directions; choose weight combiner
        both = A.multiply(AT.sign())  # keeps A_ij where AT_ij > 0
        if sym_weight == "max":
            W = both.maximum(both.T)
        elif sym_weight == "mean":
            W = (both + both.T) * 0.5
        elif sym_weight == "min":
            W = both.minimum(both.T)
        else:
            raise ValueError("sym_weight must be 'max', 'mean', or 'min'")
    else:
        raise ValueError("mode must be 'union' or 'mutual'")

    W.setdiag(0.0)
    W.eliminate_zeros()
    W.sort_indices()

    if return_distance:
        # convert similarity to distance in-place (1 - sim); keep non-negative
        W = W.tocsr(copy=False)
        W.data = 1.0 - W.data
        np.clip(W.data, 0.0, 1.0, out=W.data)

    return W#, zero_cells