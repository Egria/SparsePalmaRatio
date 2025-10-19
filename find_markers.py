#!/usr/bin/env python3
"""
Find top gene markers per label and plot distributions.

New: --plot-scope {group_vs_rest,all_cells,both}
  group_vs_rest  -> label vs rest (default; old behavior)
  all_cells      -> one global distribution per gene
  both           -> write both styles

Other plot flags unchanged:
  --plot {none,hist,kde,both}
  --kde-sample-size, --bins, --layer-for-plots, --no-log1p, --plot-top-per-label
"""

import os
import argparse
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

try:
    import scanpy as sc
except Exception as e:
    raise SystemExit(
        "This script requires Scanpy. Install with:\n"
        "  pip install scanpy anndata scipy numpy pandas scikit-learn\n"
        f"Import error: {e}"
    )


# ----------------------------- core utils -----------------------------

def _build_adata(X: sp.csr_matrix, cell_ids: pd.Index, gene_names: pd.Index,
                 labels: pd.Series, label_col: str = "label") -> "sc.AnnData":
    if not sp.isspmatrix_csr(X):
        X = sp.csr_matrix(X)
    adata = sc.AnnData(X, dtype=np.float32)
    adata.obs_names = pd.Index(cell_ids.astype(str))
    adata.var_names = pd.Index(gene_names.astype(str))
    adata.obs[label_col] = pd.Categorical(labels.reindex(adata.obs_names).astype(str))
    adata.layers["counts"] = adata.X.copy()
    return adata


def _normalize_and_log(adata: "sc.AnnData", target_sum: float = 1e4) -> None:
    sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)
    adata.layers["norm"] = adata.X.copy()
    sc.pp.log1p(adata)


def _col_means_of_layer(adata: "sc.AnnData", layer: str, mask: np.ndarray) -> np.ndarray:
    X = adata.layers[layer]
    if sp.issparse(X):
        return np.asarray(X[mask].mean(axis=0)).ravel()
    else:
        return X[mask].mean(axis=0)


def _col_nnz_of_layer(adata: "sc.AnnData", layer: str, mask: np.ndarray) -> np.ndarray:
    X = adata.layers[layer]
    if sp.issparse(X):
        return X[mask].getnnz(axis=0)
    else:
        return (X[mask] > 0).sum(axis=0)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if np.all(y_score == y_score[0]) or (y_true.min() == y_true.max()):
            return np.nan
        return float(roc_auc_score(y_true.astype(int), y_score))
    except Exception:
        return np.nan


# ----------------------------- marker finding -----------------------------

def _rank_and_augment(adata: "sc.AnnData", label_col: str, method: str = "wilcoxon",
                      n_top: int = 50, compute_auc_top_n: int = 200,
                      min_cells_per_gene: int = 3, eps: float = 1e-9) -> Dict[str, pd.DataFrame]:
    if min_cells_per_gene and min_cells_per_gene > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    _normalize_and_log(adata, target_sum=1e4)

    sc.tl.rank_genes_groups(
        adata, groupby=label_col, method=method,
        corr_method="benjamini-hochberg", n_genes=None, use_raw=False
    )

    df_all = sc.get.rank_genes_groups_df(adata, group=None)
    df_all.rename(columns={"names": "gene", "pvals": "pval", "pvals_adj": "qval"}, inplace=True)

    labels = list(adata.obs[label_col].cat.categories)
    results: Dict[str, pd.DataFrame] = {}

    counts_layer, norm_layer = "counts", "norm"
    n_cells_total = adata.n_obs
    membership = {lab: (adata.obs[label_col].values == lab) for lab in labels}

    for lab in labels:
        mask_in = membership[lab]
        mask_out = ~mask_in
        n_in, n_out = int(mask_in.sum()), int(mask_out.sum())
        if n_in == 0 or n_out == 0:
            continue

        mean_in = _col_means_of_layer(adata, norm_layer, mask_in)
        mean_out = _col_means_of_layer(adata, norm_layer, mask_out)
        det_in = _col_nnz_of_layer(adata, counts_layer, mask_in) / n_in
        det_out = _col_nnz_of_layer(adata, counts_layer, mask_out) / n_out
        log2fc = np.log2((mean_in + eps) / (mean_out + eps))

        aug = pd.DataFrame({
            "gene": adata.var_names.values,
            "mean_in_norm": mean_in.astype(np.float32),
            "mean_out_norm": mean_out.astype(np.float32),
            "pct_in": det_in.astype(np.float32),
            "pct_out": det_out.astype(np.float32),
            "log2fc": log2fc.astype(np.float32),
        })

        df_lab = df_all[df_all["group"] == lab].merge(aug, on="gene", how="left")
        df_lab = df_lab.sort_values(["qval", "scores"], ascending=[True, False], kind="mergesort")
        df_lab.insert(0, "rank", np.arange(1, len(df_lab) + 1))

        # Optional AUC
        if compute_auc_top_n and compute_auc_top_n > 0:
            y_true = mask_in.astype(int)
            top_genes = df_lab["gene"].head(compute_auc_top_n).tolist()
            gene_to_col = {g: i for i, g in enumerate(adata.var_names)}
            auc_vals = np.full(len(df_lab), np.nan, dtype=np.float32)
            X_for_auc = adata.layers[norm_layer]
            for i, g in enumerate(top_genes):
                col = gene_to_col.get(g, None)
                if col is None:
                    continue
                if sp.issparse(X_for_auc):
                    y_score = np.asarray(X_for_auc[:, col].todense()).ravel()
                else:
                    y_score = X_for_auc[:, col]
                auc_vals[i] = _safe_auc(y_true, y_score)
            df_lab["auc_norm_expr"] = auc_vals
        else:
            df_lab["auc_norm_expr"] = np.nan

        df_lab["cells_in_label"] = n_in
        df_lab["label_frequency"] = n_in / float(n_cells_total)

        if n_top is not None and n_top > 0:
            df_lab = df_lab.head(n_top).copy()

        results[lab] = df_lab.reset_index(drop=True)

    return results


def _write_outputs(per_label: Dict[str, pd.DataFrame], outdir: str) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    for lab, df in per_label.items():
        safe = _sanitize(lab)
        df.to_csv(os.path.join(outdir, f"{safe}_markers.csv"), index=False)
    cols = ["label", "rank", "gene", "scores", "pval", "qval",
            "log2fc", "mean_in_norm", "mean_out_norm", "pct_in", "pct_out",
            "auc_norm_expr", "cells_in_label", "label_frequency"]
    long_df = (pd.concat([df.assign(label=lab) for lab, df in per_label.items()],
                         axis=0, ignore_index=True)[cols]
               .sort_values(["label", "rank"]))
    long_df.to_csv(os.path.join(outdir, "markers_long_table.csv"), index=False)
    return long_df


# ----------------------------- plotting -----------------------------

def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(name))


def _get_gene_vector(adata: "sc.AnnData", gene: str, layer: str = "norm", log1p: bool = True) -> np.ndarray:
    if gene not in adata.var_names:
        raise KeyError(f"Gene '{gene}' not found")
    col = int(np.where(adata.var_names == gene)[0][0])
    X = adata.layers[layer] if layer in adata.layers else adata.X
    if sp.issparse(X):
        v = np.asarray(X[:, col].todense()).ravel()
    else:
        v = np.asarray(X[:, col]).ravel()
    return np.log1p(v) if log1p else v


def _plot_hist(vals: np.ndarray, title: str, out_png: str, bins: int = 50, xlabel: str = "log1p(normalized counts)"):
    plt.figure()
    plt.hist(vals, bins=bins, density=True, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _plot_kde(vals: np.ndarray, title: str, out_png: str, sample_for_kde: int = 20000,
              xlabel: str = "log1p(normalized counts)"):
    # Downsample if large
    if vals.size > sample_for_kde:
        idx = np.random.choice(vals.size, size=sample_for_kde, replace=False)
        vals = vals[idx]
    plt.figure()
    if np.unique(vals).size > 1:
        xs = np.linspace(vals.min(), vals.max(), 256)
        kde = gaussian_kde(vals)
        plt.plot(xs, kde(xs))
    else:
        plt.plot(vals, np.zeros_like(vals), ".")
    plt.xlabel(xlabel)
    plt.ylabel("Density (KDE)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _plot_group_vs_rest(vals_in: np.ndarray, vals_out: np.ndarray, gene: str, label: str,
                        outdir: str, bins: int, sample_for_kde: int, do_hist: bool, do_kde: bool):
    # Hist overlay
    if do_hist:
        plt.figure()
        plt.hist(vals_in, bins=bins, density=True, alpha=0.6, label=str(label))
        plt.hist(vals_out, bins=bins, density=True, alpha=0.6, label="rest")
        plt.xlabel("log1p(normalized counts)")
        plt.ylabel("Density")
        plt.title(f"{gene} — {label} vs rest (hist)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{_sanitize(gene)}_hist.png"), dpi=160)
        plt.close()
    # KDE overlay
    if do_kde:
        # Optional downsampling handled inside
        def _maybe_sample(v: np.ndarray) -> np.ndarray:
            if v.size > sample_for_kde:
                idx = np.random.choice(v.size, size=sample_for_kde, replace=False)
                return v[idx]
            return v
        xin, xout = _maybe_sample(vals_in), _maybe_sample(vals_out)
        plt.figure()
        xs_min, xs_max = float(min(xin.min(), xout.min())), float(max(xin.max(), xout.max()))
        if np.isfinite(xs_min) and np.isfinite(xs_max) and xs_min != xs_max:
            xs = np.linspace(xs_min, xs_max, 256)
            if np.unique(xin).size > 1:
                kde_in = gaussian_kde(xin); plt.plot(xs, kde_in(xs), label=str(label))
            else:
                plt.plot(xin, np.zeros_like(xin), ".", label=str(label))
            if np.unique(xout).size > 1:
                kde_out = gaussian_kde(xout); plt.plot(xs, kde_out(xs), label="rest")
            else:
                plt.plot(xout, np.zeros_like(xout), ".", label="rest")
        else:
            plt.plot(xin, np.zeros_like(xin), ".", label=str(label))
            plt.plot(xout, np.zeros_like(xout), ".", label="rest")
        plt.xlabel("log1p(normalized counts)")
        plt.ylabel("Density (KDE)")
        plt.title(f"{gene} — {label} vs rest (KDE)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{_sanitize(gene)}_kde.png"), dpi=160)
        plt.close()


def _plot_all_cells(vals_all: np.ndarray, gene: str, label: str, outdir: str,
                    bins: int, sample_for_kde: int, do_hist: bool, do_kde: bool):
    """Plots that ignore labels: global gene distribution; saved in the label's folder."""
    if do_hist:
        _plot_hist(vals_all, title=f"{gene} — all cells (hist)",
                   out_png=os.path.join(outdir, f"{_sanitize(gene)}_allcells_hist.png"),
                   bins=bins)
    if do_kde:
        _plot_kde(vals_all, title=f"{gene} — all cells (KDE)",
                  out_png=os.path.join(outdir, f"{_sanitize(gene)}_allcells_kde.png"),
                  sample_for_kde=sample_for_kde)


def _plot_markers_for_label(adata: "sc.AnnData", label_col: str, label: str, genes: List[str],
                            base_outdir: str, plot: str, bins: int, layer: str, use_log1p: bool,
                            kde_sample_size: int, scope: str):
    safe_label = _sanitize(label)
    outdir = os.path.join(base_outdir, "plots", safe_label)
    os.makedirs(outdir, exist_ok=True)

    mask_in = (adata.obs[label_col].values == label)
    mask_out = ~mask_in
    do_hist = plot in ("hist", "both")
    do_kde = plot in ("kde", "both")

    for gene in genes:
        try:
            v = _get_gene_vector(adata, gene, layer=layer, log1p=use_log1p)
        except KeyError:
            continue
        if scope in ("group_vs_rest", "both"):
            _plot_group_vs_rest(v[mask_in], v[mask_out], gene, label, outdir,
                                bins=bins, sample_for_kde=kde_sample_size,
                                do_hist=do_hist, do_kde=do_kde)
        if scope in ("all_cells", "both"):
            _plot_all_cells(v, gene, label, outdir,
                            bins=bins, sample_for_kde=kde_sample_size,
                            do_hist=do_hist, do_kde=do_kde)


def _plot_all_markers(adata: "sc.AnnData", per_label: Dict[str, pd.DataFrame], outdir: str, label_col: str,
                      plot: str, plot_top_per_label: Optional[int], bins: int, layer: str,
                      use_log1p: bool, kde_sample_size: int, scope: str):
    for label, df in per_label.items():
        genes = df["gene"].tolist()
        if plot_top_per_label is not None and plot_top_per_label > 0:
            genes = genes[:plot_top_per_label]
        _plot_markers_for_label(
            adata, label_col, label, genes, outdir,
            plot=plot, bins=bins, layer=layer, use_log1p=use_log1p,
            kde_sample_size=kde_sample_size, scope=scope
        )


# ----------------------------- public API -----------------------------

def markers_from_preprocessed(
    X_csr: sp.csr_matrix, cell_index: pd.Index, gene_index: pd.Index, labels: pd.Series,
    outdir: str = "markers_out", label_col: str = "label",
    method: str = "wilcoxon", n_top: int = 50, compute_auc_top_n: int = 200,
    min_cells_per_gene: int = 3,
    # plotting options
    plot: str = "none",                          # {"none","hist","kde","both"}
    plot_scope: str = "group_vs_rest",           # {"group_vs_rest","all_cells","both"}
    plot_top_per_label: Optional[int] = None,    # defaults to n_top
    bins: int = 50, layer_for_plots: str = "norm",
    log1p_for_plots: bool = True, kde_sample_size: int = 20000
) -> pd.DataFrame:
    adata = _build_adata(X_csr, cell_index, gene_index, labels, label_col=label_col)
    per_label = _rank_and_augment(
        adata, label_col=label_col, method=method, n_top=n_top,
        compute_auc_top_n=compute_auc_top_n, min_cells_per_gene=min_cells_per_gene
    )
    long_df = _write_outputs(per_label, outdir)

    plot = (plot or "none").lower()
    if plot != "none":
        if plot_top_per_label is None:
            plot_top_per_label = n_top
        _plot_all_markers(
            adata, per_label, outdir, label_col,
            plot=plot, plot_top_per_label=plot_top_per_label,
            bins=bins, layer=layer_for_plots, use_log1p=log1p_for_plots,
            kde_sample_size=kde_sample_size, scope=plot_scope
        )
    return long_df


def markers_from_h5ad(
    h5ad_path: str, label_col: str, outdir: str = "markers_out",
    method: str = "wilcoxon", n_top: int = 50, compute_auc_top_n: int = 200,
    min_cells_per_gene: int = 3,
    # plotting options
    plot: str = "none", plot_scope: str = "group_vs_rest",
    plot_top_per_label: Optional[int] = None,
    bins: int = 50, layer_for_plots: str = "norm",
    log1p_for_plots: bool = True, kde_sample_size: int = 20000
) -> pd.DataFrame:
    ad = sc.read_h5ad(h5ad_path, backed=None)
    if not ad.var_names.is_unique:
        ad.var_names_make_unique()
    if not ad.obs_names.is_unique:
        ad.obs_names_make_unique()
    X = ad.X.tocsr() if sp.issparse(ad.X) else sp.csr_matrix(ad.X)
    if label_col not in ad.obs:
        raise ValueError(f"'{label_col}' not found in .obs of {h5ad_path}")
    labels = ad.obs[label_col]
    return markers_from_preprocessed(
        X, ad.obs_names, ad.var_names, labels,
        outdir=outdir, label_col=label_col, method=method,
        n_top=n_top, compute_auc_top_n=compute_auc_top_n,
        min_cells_per_gene=min_cells_per_gene,
        plot=plot, plot_scope=plot_scope, plot_top_per_label=plot_top_per_label,
        bins=bins, layer_for_plots=layer_for_plots,
        log1p_for_plots=log1p_for_plots, kde_sample_size=kde_sample_size
    )
# ----------------------------- CLI -----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find top gene markers per label.")
    p.add_argument("--h5ad", type=str, help="Path to .h5ad (if using CLI mode).")
    p.add_argument("--label-col", type=str, default="cell_type",
                   help="Column in .obs with ground-truth labels.")
    p.add_argument("--outdir", type=str, default="markers_out", help="Output directory.")
    p.add_argument("--method", type=str, default="wilcoxon",
                   choices=["wilcoxon", "t-test", "t-test_overestim_var", "logreg"],
                   help="Differential test in scanpy.tl.rank_genes_groups.")
    p.add_argument("--top-n", type=int, default=50, help="How many markers to keep per label.")
    p.add_argument("--auc-top-n", type=int, default=200,
                   help="Compute ROC-AUC only for the top-N genes (speed). Set 0 to skip.")
    p.add_argument("--min-cells-per-gene", type=int, default=3,
                   help="Filter away genes expressed in fewer than this many cells.")
    return p.parse_args()


def main():
    args = _parse_args()
    if not args.h5ad:
        raise SystemExit("Please supply --h5ad PATH (or call markers_from_preprocessed from your code).")
    long_df = markers_from_h5ad(
        h5ad_path=args.h5ad,
        label_col=args.label_col,
        outdir=args.outdir,
        method=args.method,
        n_top=args.top_n,
        compute_auc_top_n=args.auc_top_n,
        min_cells_per_gene=args.min_cells_per_gene,
    )
    # Print a tiny summary
    print(f"Done. Wrote per-label CSVs and 'markers_long_table.csv' to: {os.path.abspath(args.outdir)}")
    with pd.option_context("display.max_rows", 8, "display.max_colwidth", 30):
        print(long_df.groupby('label').head(3))


if __name__ == "__main__":
    long_df = markers_from_h5ad(
        h5ad_path='data/human_meninges_development.h5ad',
        label_col='cell_type',
        outdir='markers_hmd',
        method='wilcoxon',
        n_top=10,
        compute_auc_top_n=200,
        min_cells_per_gene=3,
        plot='both',
        plot_scope='both'
    )