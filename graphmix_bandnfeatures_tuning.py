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


config_filename = "cfg/config_102580.json"
params = Parameters(config_filename)
matrix, cells, genes, labels = preprocess(params)
matrix_f, genes_f, cells_f = filter_counts(params, matrix, genes, cells, False)
gene_stats = calc_gene_stats(params, matrix_f, genes_f)
gene_stats = detrend(params, gene_stats)

#cell_types = ["Mono2","Mono1","DC1","DC6","DC4"]
#cell_types = ['Basal','Secretory']
cell_types = labels.cat.categories.tolist()

gini_ngene = [50*i for i in range(1,21)]
palma_ngene = [50*i for i in range(1,21)]
single_dict = [[0.0 for j in palma_ngene] for i in gini_ngene]

data_dic={}
for cell_type in cell_types:
    data_dic[cell_type] = copy.deepcopy(single_dict)
data_dic["ARI"] = copy.deepcopy(single_dict)
data_dic["NMI"] = copy.deepcopy(single_dict)


for i, gn in enumerate(gini_ngene):
    for j, pn in enumerate(palma_ngene):
        b5 = {"fano": 1.0}
        b4 = {"gini": 0.2, "fano": 0.8}
        b3 = {"gini": 1.0}
        b2 = {"gini": 0.2, "palma": 0.8}
        b1 = {"palma": 1.0}


        bands = [
            ("50-30", b5, 0.0, params.gene_nfeatures),
            ("30-10", b4, 0.0, params.gene_nfeatures),
            ("10-3", b3, 0.0, gn),
            ("3-1", b2, 0.0, params.gene_nfeatures),
            ("1-0.1", b1, 3.5, pn)
        ]

        graph, band_genes = make_channel_graphs(params, gene_stats, matrix_f, genes_f, labels, cells_f, cells,
                                                b5=b5, b4=b4, b3=b3, b2=b2, b1=b1,
                                                bands=bands, band_weights=[0.6, 0.0, 0.4, 0.0, 0.4])
        labels_f = generate_clusters(params, graph, cells_f)
        result = pd.DataFrame({"label":labels_f}, index=cells_f)
        tab, gt_breakdown, ari, nmi = compare_clusters_filtered(params, labels_f, labels, cells_f, cells)
        for cell_type in cell_types:
            data_dic[cell_type][i][j] = gt_breakdown.loc[cell_type,"f1"] \
                if gt_breakdown.loc[cell_type,"mapped_match"] else 0.0
        data_dic["ARI"][i][j] = ari
        data_dic["NMI"][i][j] = nmi
        print(gn,pn)



def generate_heatmap(data, fname):
    import numpy as np
    import matplotlib.pyplot as plt

    # convert to array
    arr = np.array(data, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect='auto', vmin=0.0, vmax=1.0)  # rank values

    # axis ticks & labels (rows = lower_bound, cols = upper_bound)
    ax.set_yticks(np.arange(len(gini_ngene)))
    ax.set_yticklabels([f"{v}" for v in gini_ngene])
    ax.set_ylabel("Gini n genes (row)")

    ax.set_xticks(np.arange(len(palma_ngene)))
    ax.set_xticklabels([f"{v}" for v in palma_ngene], rotation=45, ha='right')
    ax.set_xlabel("Palma n genes")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cell type F1 or NMI/ARI")

    ax.set_title("Heatmap of F1/NMI/ARI vs (gini/palma n genes)")
    fig.tight_layout()

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8)

    #plt.show()
    plt.savefig(fname)



folder = "graphmix_nfeatures_102580"
for cell_type in cell_types:
    _cell_type = cell_type.replace("/",".")
    generate_heatmap(data_dic[cell_type], f"{folder}/{_cell_type}_f1.png")
generate_heatmap(data_dic["ARI"], f"{folder}/ARI.png")
generate_heatmap(data_dic["NMI"], f"{folder}/NMI.png")



