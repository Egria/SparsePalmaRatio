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
from src.select_genes import select_genes
from src.refine.band1_refine import detect_ultrarare_B1
from src.cell_normalize import cpm_normalize_csr


config_filename = "cfg/config_159115.json"
params = Parameters(config_filename)
matrix, cells, genes, labels = preprocess(params)
matrix_f, genes_f, cells_f = filter_counts(params, matrix, genes, cells, False)

#matrix_f = cpm_normalize_csr(matrix_f)

gene_stats = calc_gene_stats(params, matrix_f, genes_f)
gene_stats = detrend(params, gene_stats)

#cell_types = ["Mono2","Mono1","DC1","DC6","DC4"]
#cell_types = ['Basal','Secretory']
cell_types = labels.cat.categories.tolist()

fano = [0.1*i for i in range(0,11)]
palma = [0.2*i for i in range(0,11)]
single_dict = [[0.0 for j in palma] for i in fano]

data_dic={}
for cell_type in cell_types:
    data_dic[cell_type] = copy.deepcopy(single_dict)
data_dic["ARI"] = copy.deepcopy(single_dict)
data_dic["NMI"] = copy.deepcopy(single_dict)



for i, f in enumerate(fano):
    for j, p in enumerate(palma):
        b5 = {"fano": 1.0}
        b4 = {"gini": 1.0}
        b3 = {"palma": 1.0}
        b2 = {"gini": 0.45, "palma": 0.55}
        b1 = {"gini":0.6, "palma": 0.4}

        bands = [
            ("50-30", b5, 0.0, 1000),
            ("30-10", b4, 0.0, 1000),
            ("10-3", b3, 3.5, 1000),
            ("3-1", b2, 1.0, 1000),
            ("1-0.1", b1, 3.5, 1000)
        ]

        graph, band_genes = make_channel_graphs(params, gene_stats, matrix_f, genes_f, labels, cells_f, cells,
                                                b5=b5, b2=b2, b1=b1, band_weights=[f, 1.0-f, p, 0.0, 0.0], bands=bands)
        labels_f = generate_clusters(params, graph, cells_f)
        _labels, report = detect_rare_B2(matrix_f, genes_f, labels_f, band_genes[3],
                                         A_global=graph, output_path=params.output_folder, conn_min=0.6, stab_min=0.5,
                                         random_state=12277, n_pcs=500, k_knn=500, size_min_frac_parent=0.005,
                                         mix_alpha=0.8)
        __labels, report = detect_ultrarare_B1(matrix_f, genes_f, _labels, band_genes[4],
                                               output_path=params.output_folder, conn_min=0.6, stab_min=0.5)
        result = pd.DataFrame({"label":labels_f}, index=cells_f)
        tab, gt_breakdown, ari, nmi = compare_clusters_filtered(params, __labels, labels, cells_f, cells)
        for cell_type in cell_types:
            data_dic[cell_type][i][j] = gt_breakdown.loc[cell_type,"f1"] \
                if gt_breakdown.loc[cell_type,"mapped_match"] else 0.0
        data_dic["ARI"][i][j] = ari
        data_dic["NMI"][i][j] = nmi
        print(f,p, ari, nmi)



def generate_heatmap(data, fname):
    import numpy as np
    import matplotlib.pyplot as plt

    # convert to array
    arr = np.array(data, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect='auto', vmin=0.0, vmax=1.0)  # rank values

    # axis ticks & labels (rows = lower_bound, cols = upper_bound)
    ax.set_yticks(np.arange(len(fano)))
    ax.set_yticklabels([f"{v:.1f}" for v in fano])
    ax.set_ylabel("Fano(Gini=1-Fano) graph weight (row)")

    ax.set_xticks(np.arange(len(palma)))
    ax.set_xticklabels([f"{v:.1f}" for v in palma], rotation=45, ha='right')
    ax.set_xlabel("Palma graph weight (column)")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cell type F1 or NMI/ARI")

    ax.set_title("Heatmap of F1/NMI/ARI vs (fano(gini) weight/palma weight)")
    fig.tight_layout()

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8)

    #plt.show()
    plt.savefig(fname)



folder = "graphmix_159115"
for cell_type in cell_types:
    _cell_type = cell_type.replace("/",".")
    generate_heatmap(data_dic[cell_type], f"{folder}/{_cell_type}_f1.png")
generate_heatmap(data_dic["ARI"], f"{folder}/ARI.png")
generate_heatmap(data_dic["NMI"], f"{folder}/NMI.png")