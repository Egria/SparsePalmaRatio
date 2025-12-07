#!/usr/bin/env python3
"""
Plot markers vs. gene-level metrics:
  y-axis  = quantile of a chosen metric across all genes (e.g., 'palma_final')
  x-axis  = -log10(label_frequency) for the marker's label

Inputs
------
1) metrics CSV (rows = genes; columns = one or more metrics; e.g. palma_final,gini_final,...)
2) markers CSV  (long table with at least columns: ['gene','label','label_frequency'])

Outputs
-------
- For each metric present:
    - PNG scatter plot:  <outdir>/markers_vs_<metric>_quantile.png
    - CSV of plotted points: <outdir>/<metric>_markers_scatter_data.csv
"""

#!/usr/bin/env python3
"""
Plot markers vs metrics.

y-axis: quantile of the metric across all genes
x-axis: -log10(label_frequency)

Now supports:
  --mode combined  (overlay multiple metrics in one plot, default)
  --mode separate  (old behavior: one plot per metric)
  --mode both      (do both)

Also writes a combined tidy CSV with all points.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def load_markers_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"gene", "label", "label_frequency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Markers CSV missing columns: {sorted(missing)}")
    df["gene"] = df["gene"].astype(str)
    return df


def compute_quantiles(s: pd.Series) -> pd.Series:
    """Return quantiles in [0,1]; higher values -> higher quantiles."""
    return s.rank(pct=True, method="average")


def build_scatter_table(metrics_df: pd.DataFrame,
                        markers_df: pd.DataFrame,
                        metric: str,
                        min_freq: float = 1e-12) -> pd.DataFrame:
    if metric not in metrics_df.columns:
        raise KeyError(f"Metric '{metric}' not in metrics table.")
    q = compute_quantiles(metrics_df[metric]).rename("metric_quantile")
    joined = markers_df.merge(
        pd.concat([metrics_df[[metric]], q], axis=1),
        left_on="gene", right_index=True, how="left"
    ).dropna(subset=[metric, "metric_quantile"]).copy()

    freq = joined["label_frequency"].astype(float).clip(lower=min_freq)
    joined["neg_log10_freq"] = -np.log10(freq)
    joined = joined.rename(columns={metric: "metric_value"})
    return joined[["gene", "label", "label_frequency", "neg_log10_freq",
                   "metric_value", "metric_quantile"]]


def build_combined_table(metrics_df: pd.DataFrame,
                         markers_df: pd.DataFrame,
                         metrics: list,
                         min_freq: float = 1e-12,
                         max_points_per_metric: int | None = None,
                         random_state: int = 0) -> pd.DataFrame:
    """Return a long table with a 'metric' column and data for all requested metrics."""
    rng = np.random.default_rng(random_state)
    frames = []
    for m in metrics:
        if m not in metrics_df.columns:
            continue
        t = build_scatter_table(metrics_df, markers_df, m, min_freq=min_freq)
        t.insert(0, "metric", m)
        if max_points_per_metric and len(t) > max_points_per_metric:
            idx = rng.choice(len(t), size=max_points_per_metric, replace=False)
            t = t.iloc[np.sort(idx)]
        frames.append(t)
    if not frames:
        raise ValueError("None of the requested metrics were found in the metrics table.")
    return pd.concat(frames, axis=0, ignore_index=True)


def make_combined_scatter_plot(df: pd.DataFrame, out_png: str,
                               title: str = "Markers: metric quantile vs rarity",
                               alpha: float = 0.6, size: float = 14.0) -> None:
    """Overlay multiple metrics in one axes (matplotlib will auto-assign colors)."""
    plt.figure()
    for metric, g in df.groupby("metric"):
        plt.scatter(g["neg_log10_freq"], g["metric_quantile"], s=size, alpha=alpha, label=metric)
    plt.xlabel("-log10(label_frequency)")
    plt.ylabel("Metric quantile across genes")
    plt.title(title)
    plt.legend(title="metric")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def make_separate_scatter_plot(df: pd.DataFrame, metric: str, out_png: str) -> None:
    """Single-metric scatter (legacy style)."""
    plt.figure()
    plt.scatter(df["neg_log10_freq"], df["metric_quantile"], s=18, alpha=0.7)
    plt.xlabel("-log10(label_frequency)")
    plt.ylabel(f"{metric} quantile across genes")
    plt.title(f"Markers: {metric} quantile vs rarity")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot markers vs metric quantiles.")
    p.add_argument("--metrics-csv", required=True, help="CSV with gene rows and metric columns.")
    p.add_argument("--markers-csv", required=True, help="Markers long table CSV (gene,label,label_frequency).")
    p.add_argument("--metrics", default="palma_final,gini_final,fano_final,theil_final,idf_final",
                   help="Comma-separated metric columns to plot (only those present will be used).")
    p.add_argument("--outdir", default="markers_vs_metrics_plots", help="Output directory.")
    p.add_argument("--min-freq", type=float, default=1e-12,
                   help="Lower bound to avoid log10(0) when computing -log10(label_frequency).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    metrics_df = load_metrics_table(args.metrics_csv)
    markers_df = load_markers_table(args.markers_csv)

    requested = [m.strip() for m in args.metrics.split(",") if m.strip()]
    existing = [m for m in requested if m in metrics_df.columns]
    missing = [m for m in requested if m not in metrics_df.columns]

    if missing:
        print(f"[warn] Skipping missing metrics: {', '.join(missing)}")

    for metric in existing:
        scatter_df = build_scatter_table(metrics_df, markers_df, metric, min_freq=args.min_freq)
        # write the data used for plotting (handy for downstream analyses)
        out_csv = os.path.join(args.outdir, f"{metric}_markers_scatter_data.csv")
        scatter_df.to_csv(out_csv, index=False)

        out_png = os.path.join(args.outdir, f"markers_vs_{metric}_quantile.png")
        make_separate_scatter_plot(scatter_df, metric, out_png)
        print(f"[ok] Wrote: {out_png}")
        print(f"[ok] Wrote: {out_csv}")


if __name__ == "__main__":
    metrics_csv = "./result_gse159115/gene_stats_detrend.csv"
    markers_csv = "./markers_159115/markers_long_table.csv"
    metrics = "palma_final,gini_final"
    min_freq = 1e-18
    outdir = "markers_159115"
    max_points_per_metric = None
    random_seed = 42

    metrics_df = load_metrics_table(metrics_csv)
    markers_df = load_markers_table(markers_csv)

    requested = [m.strip() for m in metrics.split(",") if m.strip()]
    existing = [m for m in requested if m in metrics_df.columns]
    missing = [m for m in requested if m not in metrics_df.columns]


    if missing:
        print(f"[warn] Skipping missing metrics: {', '.join(missing)}")

    combined = build_combined_table(
        metrics_df, markers_df, existing,
        min_freq=min_freq,
        max_points_per_metric=max_points_per_metric,
        random_state=random_seed
    )
    combined_csv = os.path.join(outdir, "combined_markers_scatter_data.csv")
    combined.to_csv(combined_csv, index=False)
    out_png = os.path.join(outdir, "markers_vs_metrics_combined.png")
    make_combined_scatter_plot(combined, out_png)
    print(f"[ok] Combined plot: {out_png}")
    print(f"[ok] Combined CSV:  {combined_csv}")
