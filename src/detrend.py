import os
import pandas as pd
import numpy as np
from .parameters import Parameters
from .detrending.gini_lowess import lowess_twopass_detrending
from .detrending.palma_mad import detrend_palma_cont_scale, detrend_palma_2pass_isotonic_stable
from .detrending.lq_linear import leastsq_linear_detrending

def _rankp(col: pd.Series) -> np.ndarray:
    G = len(col)
    p_df = (col.rank(axis=0, method='average', ascending=True) - 0.5) / G
    return p_df.to_numpy()

def detrend(params: Parameters,gene_stats: pd.DataFrame) -> pd.DataFrame:
    log2max = np.asarray(gene_stats['log2max'], dtype=float)
    gini = np.asarray(gene_stats['gini'], dtype=float)
    palma = np.asarray(gene_stats['palma'], dtype=float)
    theil = np.asarray(gene_stats['theil'], dtype=float)
    #idf = np.asarray(gene_stats['idf'], dtype=float)
    coverage = np.asarray(gene_stats['coverage'], dtype=float)
    gini_d = lowess_twopass_detrending(log2max, gini)
    palma_r, palma_z = detrend_palma_2pass_isotonic_stable(log2max, palma, params.mad_eps,
                                                           params.mad_topcut, params.mad_nbins)
    palma_d = lowess_twopass_detrending(log2max, palma)
    theil_d = lowess_twopass_detrending(log2max, theil)
    coverage_d = lowess_twopass_detrending(log2max, coverage)
    palma_q = leastsq_linear_detrending(log2max, palma)
    gene_stats['gini_final'] = _rankp(pd.Series(gini_d))
    gene_stats['palma_final'] = _rankp(pd.Series(palma_d))
    gene_stats['fano_final'] = _rankp(gene_stats['fano'])
    gene_stats['theil_final'] = _rankp(pd.Series(theil_d))
    gene_stats['idf_final'] = _rankp(gene_stats['idf'])
    gene_stats['coverage_final'] = _rankp(gene_stats['coverage'])
    gene_stats['palma_r'] = palma_r
    gene_stats['palma_z'] = palma_z
    gene_stats['palma_d'] = palma_d
    gene_stats['palma_q'] = palma_q

    csv_path = os.path.join(params.output_folder, "gene_stats_detrend.csv")
    gene_stats.to_csv(csv_path,index=True, header=True)
    return gene_stats
