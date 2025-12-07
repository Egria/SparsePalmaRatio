import scipy.sparse as sp
import pandas as pd
import numpy as np
from .parameters import Parameters
from .refine.band2_refine import detect_rare_B2, detect_rare_B2_new
from .refine.band1_refine import detect_ultrarare_B1, detect_ultrarare_B1_new
from .refine.band1_harvest import augment_B1_children_from_labels

def refine_cluster(
        params: Parameters,
        mat: sp.csr_matrix,
        genes,
        labels,
        band_genes,
        global_graph
):
    _labels, report = detect_rare_B2(mat, genes, labels, band_genes[3],A_global=global_graph,
                                     output_path=params.output_folder, conn_min=0.3, stab_min=0.75,
                                     random_state=12277, n_pcs=20, k_knn=20, size_min_frac_parent=0.005, mix_alpha=0.7,n_markers_min=1,auc_min=0.78)
    labels_f, report = detect_ultrarare_B1(mat, genes, _labels, band_genes[4], output_path=params.output_folder, conn_min=0.5, stab_min=0.4, n_markers_min=1)
    #labels_f, report = augment_B1_children_from_labels(mat, genes, _labels, band_genes[4],
    #                                                   output_path=params.output_folder)
    return labels_f

def refine_cluster_new(
    params,
    mat: sp.csr_matrix,
    genes_index: pd.Index,
    labels: np.ndarray,
    global_graph: sp.csr_matrix,
    lowess_fun
):
    """
    New refine pipeline:
      1) B2: local Palma (top-500) + 2-pass lowess per parent (major),
      2) B1: local Palma (top-500) + 2-pass lowess per parent (after B2).
    """
    # B2 refine
    _labels, report_b2 = detect_rare_B2_new(
        X_gc=mat,
        genes_index=genes_index,
        labels_major=labels,
        use_arctan=False,
        n_pcs=20,
        k_knn=20,
        resolution=1.5,
        A_global=global_graph,
        mix_alpha=0.7,
        conn_min=0.2,
        auc_min=0.80,
        n_markers_min=1,
        stab_bootstrap=20,
        stab_frac=0.8,
        stab_min=0.4,
        random_state=12277,
        output_path=params.output_folder,
        top_n_genes=500,
        palma_top=0.1,
        palma_bottom=0.8,
        palma_alpha=1e-5,
        lowess_fun=lowess_fun
    )

    # B1 refine on B2-refined parents
    labels_f, report_b1 = detect_ultrarare_B1_new(
        X_gc=mat,
        genes_index=genes_index,
        labels_major=_labels,
        use_arctan=False,
        n_pcs=15,
        k_knn=20,
        conn_min=0.5,
        auc_min=0.85,
        n_markers_min=0,
        stab_bootstrap=20,
        stab_frac=0.8,
        stab_min=0.2,
        random_state=12277,
        output_path=params.output_folder,
        top_n_genes=500,
        palma_top=0.1,
        palma_bottom=0.8,
        palma_alpha=1e-5,
        lowess_fun=lowess_fun
    )

    return labels_f