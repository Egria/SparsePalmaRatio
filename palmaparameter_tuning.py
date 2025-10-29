import pandas as pd
import copy

from src.parameters import Parameters
from src.preprocess import preprocess
from src.filter import filter_counts
from src.calc_stat import calc_gene_stats
from src.detrend import detrend
from src.generate_cluster import generate_clusters
from src.comparison import compare_clusters_filtered
from src.channels import make_channel_graphs
from src.refine_cluster import refine_cluster


def generate_heatmap(data, fname):
    import numpy as np
    import matplotlib.pyplot as plt

    # convert to array
    arr = np.array(data, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect='auto')  # rank values

    # axis ticks & labels (rows = lower_bound, cols = upper_bound)
    ax.set_yticks(np.arange(len(lower_bound)))
    ax.set_yticklabels([str(v) for v in lower_bound])
    ax.set_ylabel("palma_lower (row)")

    ax.set_xticks(np.arange(len(upper_bound)))
    ax.set_xticklabels([str(v) for v in upper_bound], rotation=45, ha='right')
    ax.set_xlabel("palma_upper (column)")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("palma_d rank (lower is better)")

    ax.set_title("Heatmap of palma_d rank vs (lower_bound, upper_bound)")
    fig.tight_layout()

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.0f}", ha="center", va="center", fontsize=8)

    #plt.show()
    plt.savefig(fname)


config_filename = "cfg/config_hmd.json"
markers = ["ENSG00000116729","ENSG00000111783","ENSG00000173372","ENSG00000173369","ENSG00000223609","ENSG00000111348",
           "ENSG00000187514","ENSG00000077420","ENSG00000108018","ENSG00000188517","ENSG00000137285","ENSG00000139910",
           "ENSG00000173267","ENSG00000277586","ENSG00000173068","ENSG00000152661","ENSG00000075884","ENSG00000111348"]
params = Parameters(config_filename)
matrix, cells, genes, labels = preprocess(params)
matrix_f, genes_f, cells_f = filter_counts(params, matrix, genes, cells, False)

lower_bound = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
               0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
upper_bound = [0.1, 0.05, 0.02, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]

palma_d_rank = [[99999 for j in upper_bound] for i in lower_bound]
palma_r_rank = [[99999 for j in upper_bound] for i in lower_bound]
palma_z_rank = [[99999 for j in upper_bound] for i in lower_bound]

data_dic={}
for marker in markers:
    data_dic[marker] = []
    data_dic[marker].append(copy.deepcopy(palma_d_rank))
    data_dic[marker].append(copy.deepcopy(palma_r_rank))
    data_dic[marker].append(copy.deepcopy(palma_z_rank))


for i, lb in enumerate(lower_bound):
    for j, ub in enumerate(upper_bound):
        params.palma_lower = lb
        params.palma_upper = ub
        gene_stats = calc_gene_stats(params, matrix_f, genes_f)
        gene_stats = detrend(params, gene_stats)
        for marker in markers:
            rd = gene_stats["palma_d"].rank(method='average', ascending=False).loc[marker]
            rr = gene_stats["palma_r"].rank(method='average', ascending=False).loc[marker]
            rz = gene_stats["palma_z"].rank(method='average', ascending=False).loc[marker]
            data_dic[marker][0][i][j] = rd
            data_dic[marker][1][i][j] = rr
            data_dic[marker][2][i][j] = rz
            print(lb, ub, marker)


folder = "heatmap_hmd"
for marker in markers:
    generate_heatmap(data_dic[marker][0], f"{folder}/{marker}_palma_d.png")
    generate_heatmap(data_dic[marker][1], f"{folder}/{marker}_palma_r.png")
    generate_heatmap(data_dic[marker][2], f"{folder}/{marker}_palma_z.png")













