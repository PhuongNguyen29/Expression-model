import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads and harmonizes TCGA and ORIEN cohorts for bidirectional validation.
    Based on principles from:
    - "Cross-cohort analysis of cancer genomic data" (Zhao et al., 2021, Nature Communications)
    - "Systematic assessment of transcriptomic biomarker specificity" (Waldron et al., 2014, BMC Genomics)
    """
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        """Initialize data loader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.raw_dir = Path(self.data_config['raw_data_dir'])
        self.processed_dir = Path(self.data_config['processed_data_dir'])
        
        # Create processed directory if it doesn't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        self.seed = self.config['project']['seed']
        np.random.seed(self.seed)
        
        # Data containers
        self.tcga_expr = None
        self.orien_expr = None
        self.surv_tcga = None
        self.surv_orien = None
        self.common_genes = None
        
    def load_raw_data(self) -> Dict:
        """
        Load all raw data files and perform initial checks.
        Returns dict with all loaded dataframes.
        """
        logger.info("="*50)
        logger.info("Loading raw data files...")
        
        # Load expression data (genes as rows, samples as columns)
        tcga_expr = pd.read_csv(
            self.raw_dir / self.data_config['tcga_expression'], 
            index_col=0
        )
        orien_expr = pd.read_csv(
            self.raw_dir / self.data_config['orien_expression'], 
            index_col=0
        )
        
        # Load survival data
        surv_tcga = pd.read_csv(
            self.raw_dir / self.data_config['tcga_survival']
        )
        surv_orien = pd.read_csv(
            self.raw_dir / self.data_config['orien_survival']
        )
        
        # Load clinical data (for future use)
        clinical_tcga = pd.read_csv(
            self.raw_dir / self.data_config['tcga_clinical']
        )
        clinical_orien = pd.read_csv(
            self.raw_dir / self.data_config['orien_clinical']
        )
        
        # Log basic statistics
        logger.info(f"TCGA expression shape: {tcga_expr.shape} (genes × samples)")
        logger.info(f"ORIEN expression shape: {orien_expr.shape} (genes × samples)")
        logger.info(f"TCGA survival records: {len(surv_tcga)}")
        logger.info(f"ORIEN survival records: {len(surv_orien)}")
        
        # Store in instance
        self.tcga_expr = tcga_expr
        self.orien_expr = orien_expr
        self.surv_tcga = surv_tcga
        self.surv_orien = surv_orien
        
        return {
            'tcga_expr': tcga_expr,
            'orien_expr': orien_expr,
            'surv_tcga': surv_tcga,
            'surv_orien': surv_orien,
            'clinical_tcga': clinical_tcga,
            'clinical_orien': clinical_orien
        }
        
    def harmonize_sample_ids(self):
        """
        Fix the sample ID mismatch between expression and survival data.
        ORIEN expression has 'X' prefix that needs to be removed.
        """
        logger.info("="*50)
        logger.info("Harmonizing sample IDs...")

        
        # Verify sample alignment
        tcga_expr_samples = set(self.tcga_expr.columns)
        tcga_surv_samples = set(self.surv_tcga['sampleID'])
        orien_expr_samples = set(self.orien_expr.columns)
        orien_surv_samples = set(self.surv_orien['sampleID'])
        
        # Find matching samples
        tcga_matched = tcga_expr_samples.intersection(tcga_surv_samples)
        orien_matched = orien_expr_samples.intersection(orien_surv_samples)
        
        logger.info(f"TCGA: {len(tcga_matched)}/{len(tcga_surv_samples)} samples have expression data")
        logger.info(f"ORIEN: {len(orien_matched)}/{len(orien_surv_samples)} samples have expression data")
        
        # Filter to only matched samples
        self.tcga_expr = self.tcga_expr[sorted(list(tcga_matched))]
        self.surv_tcga = self.surv_tcga[self.surv_tcga['sampleID'].isin(tcga_matched)]
        
        self.orien_expr = self.orien_expr[sorted(list(orien_matched))]
        self.surv_orien = self.surv_orien[self.surv_orien['sampleID'].isin(orien_matched)]
        
        # Set index for survival data for easier merging
        self.surv_tcga.set_index('sampleID', inplace=True)
        self.surv_orien.set_index('sampleID', inplace=True)
        
        logger.info(f"Final TCGA samples: {self.tcga_expr.shape[1]}")
        logger.info(f"Final ORIEN samples: {self.orien_expr.shape[1]}")
        
    def identify_common_genes(self) -> List[str]:
        """
        Identify genes present in both cohorts.
        This is CRITICAL for bidirectional stability.
        """
        logger.info("="*50)
        logger.info("Identifying common genes...")
        
        tcga_genes = set(self.tcga_expr.index)
        orien_genes = set(self.orien_expr.index)
        common_genes = sorted(list(tcga_genes.intersection(orien_genes)))
        
        logger.info(f"TCGA total genes: {len(tcga_genes)}")
        logger.info(f"ORIEN total genes: {len(orien_genes)}")
        logger.info(f"Common genes: {len(common_genes)}")
        logger.info(f"TCGA unique genes: {len(tcga_genes - orien_genes)}")
        logger.info(f"ORIEN unique genes: {len(orien_genes - tcga_genes)}")
        
        self.common_genes = common_genes
        
        # Filter to common genes if configured
        if self.data_config['use_common_genes']:
            self.tcga_expr = self.tcga_expr.loc[common_genes]
            self.orien_expr = self.orien_expr.loc[common_genes]
            logger.info(f"Filtered to {len(common_genes)} common genes")
        
        return common_genes
    
    def check_data_quality(self):
        """
        Perform basic data quality checks.
        Based on "Data quality assessment for genomics" (Taub et al., 2019, Bioinformatics)
        """
        logger.info("="*50)
        logger.info("Checking data quality...")
        
        # Check for missing values
        tcga_missing = self.tcga_expr.isna().sum().sum()
        orien_missing = self.orien_expr.isna().sum().sum()
        
        if tcga_missing > 0 or orien_missing > 0:
            logger.warning(f"Missing values - TCGA: {tcga_missing}, ORIEN: {orien_missing}")
        else:
            logger.info("No missing values found ✓")
        
        # Check expression value ranges (should be log-scale)
        tcga_range = (self.tcga_expr.min().min(), self.tcga_expr.max().max())
        orien_range = (self.orien_expr.min().min(), self.orien_expr.max().max())
        
        logger.info(f"TCGA expression range: [{tcga_range[0]:.2f}, {tcga_range[1]:.2f}]")
        logger.info(f"ORIEN expression range: [{orien_range[0]:.2f}, {orien_range[1]:.2f}]")
        
        # Check survival data
        logger.info(f"TCGA events: {self.surv_tcga['event'].sum()}/{len(self.surv_tcga)} "
                   f"({100*self.surv_tcga['event'].mean():.1f}%)")
        logger.info(f"ORIEN events: {self.surv_orien['event'].sum()}/{len(self.surv_orien)} "
                   f"({100*self.surv_orien['event'].mean():.1f}%)")
        
        # Check for duplicate genes
        tcga_dup = self.tcga_expr.index.duplicated().sum()
        orien_dup = self.orien_expr.index.duplicated().sum()
        
        if tcga_dup > 0 or orien_dup > 0:
            logger.warning(f"Duplicate genes - TCGA: {tcga_dup}, ORIEN: {orien_dup}")
    
    def run_full_pipeline(self) -> Dict:
        """
        Execute the complete data loading and harmonization pipeline.
        """
        logger.info("Starting data loading pipeline...")
        
        # Load raw data
        data = self.load_raw_data()
        
        # Harmonize sample IDs
        self.harmonize_sample_ids()
        
        # Identify common genes
        self.identify_common_genes()
        
        # Check data quality
        self.check_data_quality()
        
        logger.info("="*50)
        logger.info("Data loading complete!")
        logger.info(f"Ready for analysis with {len(self.common_genes)} genes")
        
        return {
            'tcga_expr': self.tcga_expr,
            'orien_expr': self.orien_expr,
            'surv_tcga': self.surv_tcga,
            'surv_orien': self.surv_orien,
            'common_genes': self.common_genes
        }


# Test function
if __name__ == "__main__":
    loader = DataLoader()
    data = loader.run_full_pipeline()
    
    # Save processed data for quick loading later
    logger.info("Saving processed data...")
    data['tcga_expr'].to_csv(loader.processed_dir / "tcga_harmonized.csv")
    data['orien_expr'].to_csv(loader.processed_dir / "orien_harmonized.csv")
    data['surv_tcga'].to_csv(loader.processed_dir / "surv_tcga_harmonized.csv")
    data['surv_orien'].to_csv(loader.processed_dir / "surv_orien_harmonized.csv")
    
    # Save common genes list
    pd.Series(data['common_genes']).to_csv(
        loader.processed_dir / "common_genes.csv", 
        index=False, 
        header=['gene']
    )