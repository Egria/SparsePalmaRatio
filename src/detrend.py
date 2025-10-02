import os
import pandas as pd
import numpy as np
from .parameters import Parameters
from .detrending.gini_lowess import lowess_twopass_detrending
from .detrending.palma_mad import detrend_palma_2pass_isotonic

def detrend(params: Parameters,
            genes_f,
            log2max: np.ndarray,
            gini: np.ndarray,
            palma: np.ndarray):
    gini_d = lowess_twopass_detrending(log2max, gini)
    palma_r, palma_z = detrend_palma_2pass_isotonic(log2max, palma, params.mad_eps, params.mad_topcut, params.mad_nbins)


    csv_path = os.path.join(params.output_folder, "gene_stats_detrend.csv")
    pd.DataFrame({"gini": gini, "gini_fit2": gini_d, "palma": palma, "palma_r2": palma_r,
                   "log2max": log2max}, index=genes_f).to_csv(csv_path,index=True, header=True)
    return gini_d, palma_r
