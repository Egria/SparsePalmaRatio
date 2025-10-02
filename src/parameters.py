import json
class Parameters:
    def __init__(self, config_filename:str):
        with open(config_filename,"r") as f:
            config = json.load(f)
        self.raw_filename: str = config.get("raw_filename", "")
        self.output_folder: str = config.get("output_folder", "result")
        self.cell_type_column: str = config.get("cell_type_column", "cell_type")
        self.gene_name_column:str = config.get("gene_name_column", "Gene")
        self.min_cells: int = config.get("min_cells", 3)
        self.min_genes: int = config.get("min_genes", 1000)
        self.expression_cutoff: int = config.get("expression_cutoff", 0)
        self.log2_cutoffl: float = config.get("log2_cutoffl", 0.0)
        self.log2_cutoffh: float = config.get("log2_cutoffh", 20.0)
        self.palma_alpha: float = config.get("palma_alpha", 1e-6)
        self.palma_upper: float = config.get("palma_upper", 0.1)
        self.palma_lower: float = config.get("palma_lower", 0.4)
        self.palma_winsor: float = config.get("palma_winsor", 0.0)
        self.mad_eps:float = config.get("mad_eps", 1e-8)
        self.mad_nbins:float = config.get("mad_nbins", 401)
        self.mad_topcut:float = config.get("mad_topcut", 0.01)
        self.select_standard:str = config.get("select_standard", "pval")
        self.gene_topcut:int | float = config.get("gene_topcut", 50)
        self.pval_threshold:float = config.get("pval_threshold", 1e-4)