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
from src.refine.band2_refine import detect_rare_B2


config_filename = "cfg/config_102580.json"
params = Parameters(config_filename)
matrix, cells, genes, labels = preprocess(params)
matrix_f, genes_f, cells_f = filter_counts(params, matrix, genes, cells, False)
gene_stats = calc_gene_stats(params, matrix_f, genes_f)
gene_stats = detrend(params, gene_stats)





#cell_types = ["Mono2","Mono1","DC1","DC6","DC4"]
#cell_types = ['Basal','Secretory']
cell_types = labels.cat.categories.tolist()

mix_alpha = [0.05*i for i in range(0,21)]
size_min = [0.0005*i for i in range(0,21)]
single_dict = [[0.0 for j in size_min] for i in mix_alpha]

data_dic={}
for cell_type in cell_types:
    data_dic[cell_type] = copy.deepcopy(single_dict)
data_dic["ARI"] = copy.deepcopy(single_dict)
data_dic["NMI"] = copy.deepcopy(single_dict)

overlap = None
pcan = 500
b5 = {"fano": 1.0}
b4 = {"gini": 0.2, "fano": 0.8}
b3 = {"gini": 1.0}
b2 = {"gini": 0.45, "palma": 0.55}
b1 = {"palma": 1.0}

bands = [
    ("50-30", b5, 0.0, 500),
    ("30-10", b4, 0.0, params.gene_nfeatures),
    ("10-3", b3, 0.0, 50),
    ("3-1", b2, 1.0, 950),
    ("1-0.1", b1, 3.5, 850)
]
#print(b2n)

graph, band_genes = make_channel_graphs(params, gene_stats, matrix_f, genes_f, labels, cells_f, cells,
                                        b5=b5, b3=b3, b4=b4, b2=b2, b1=b1, bands=bands,
                                        band_weights=[0.8, 0.0, 0.2, 0.0, 0.6])
labels_f = generate_clusters(params, graph, cells_f)

for i, alpha in enumerate(mix_alpha):
    for j, smin in enumerate(size_min):
        _labels, report = detect_rare_B2(matrix_f, genes_f, labels_f, band_genes[3],
                                         A_global=graph, output_path=params.output_folder, conn_min=0.6, stab_min=0.5,
                                         random_state=12277, n_pcs=500, k_knn=500, size_min_frac_parent=smin, mix_alpha=alpha)
        tab, gt_breakdown, ari, nmi = compare_clusters_filtered(params, _labels, labels, cells_f, cells)
        for cell_type in cell_types:
            data_dic[cell_type][i][j] = gt_breakdown.loc[cell_type,"f1"] \
                if gt_breakdown.loc[cell_type,"mapped_match"] else 0.0
        data_dic["ARI"][i][j] = ari
        data_dic["NMI"][i][j] = nmi
        print(alpha, smin, ari, nmi)


def generate_heatmap(data, fname):
    import numpy as np
    import matplotlib.pyplot as plt

    # convert to array
    arr = np.array(data, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect='auto', vmin=0.0, vmax=1.0)  # rank values

    # axis ticks & labels (rows = lower_bound, cols = upper_bound)
    ax.set_yticks(np.arange(len(mix_alpha)))
    ax.set_yticklabels([f"{v:.2f}" for v in mix_alpha])
    ax.set_ylabel("Graph Mix alpha of Band2 features (row)")

    ax.set_xticks(np.arange(len(size_min)))
    ax.set_xticklabels([f"{100*v :2f}" for v in size_min], rotation=45, ha='right')
    ax.set_xlabel("B2 size min frac (column)")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cell type F1 or NMI/ARI")

    ax.set_title("Heatmap of F1/NMI/ARI vs (B2 refinement Mix alpha/Size min)")
    fig.tight_layout()

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8)

    #plt.show()
    plt.savefig(fname)



folder = "b2refinement_pca_102580"
for cell_type in cell_types:
    _cell_type = cell_type.replace("/",".")
    generate_heatmap(data_dic[cell_type], f"{folder}/{_cell_type}_f1.png")
generate_heatmap(data_dic["ARI"], f"{folder}/ARI.png")
generate_heatmap(data_dic["NMI"], f"{folder}/NMI.png")



