import scipy.sparse as sp
from .parameters import Parameters
from .refine.band2_refine import detect_rare_B2
from .refine.band1_refine import detect_ultrarare_B1
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
                                     output_path=params.output_folder, conn_min=0.4, stab_min=0.5)
    _labels, report = detect_ultrarare_B1(mat, genes, _labels, band_genes[4], output_path=params.output_folder)
    labels_f, report = augment_B1_children_from_labels(mat, genes, _labels, band_genes[4],
                                                       output_path=params.output_folder)
    return labels_f