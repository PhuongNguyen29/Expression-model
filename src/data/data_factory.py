"""
Dataset Factory for Flexible Data Loading
Supports multiple dataset types: full_genes, iqr_filtered, biomarker, pathway-based
Based on Factory Design Pattern (Gang of Four, 1994)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DatasetFactory:
    """
    Factory for creating different dataset configurations.
    
    Supported dataset types:
    - full_genes: All common genes (~14,778)
    - iqr_filtered: IQR-filtered genes
    - biomarker: Curated biomarker genes
    - pathway: Pathway-based gene sets (future)
    """
    
    SUPPORTED_DATASETS = ['full_genes', 'iqr_filtered', 'biomarker', 'pathway']
    
    def __init__(self, config: Dict):
        """
        Initialize factory with configuration.
        
        Args:
            config: Experiment configuration dict with 'dataset' and 'paths' keys
        """
        self.config = config
        self.dataset_config = config['dataset']
        self.paths_config = config['paths']
        
        self.raw_dir = Path(self.paths_config['raw_data_dir'])
        self.processed_dir = Path(self.paths_config['processed_data_dir'])
        
        # Validate dataset type
        dataset_name = self.dataset_config['name']
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset type: {dataset_name}. "
                f"Supported types: {self.SUPPORTED_DATASETS}"
            )
    
    def load_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Load the configured dataset.
        
        Returns:
            dict with keys: 'tcga_expr', 'orien_expr', 'surv_tcga', 'surv_orien'
        """
        dataset_name = self.dataset_config['name']
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load expression data based on config
        tcga_expr_file = self.raw_dir / self.dataset_config['tcga_expression']
        orien_expr_file = self.raw_dir / self.dataset_config['orien_expression']
        
        logger.info(f"  TCGA expression: {tcga_expr_file.name}")
        logger.info(f"  ORIEN expression: {orien_expr_file.name}")
        
        tcga_expr = pd.read_csv(tcga_expr_file, index_col=0)
        orien_expr = pd.read_csv(orien_expr_file, index_col=0)
        
        # Load survival data (always the same files)
        surv_tcga = pd.read_csv(
            self.raw_dir / self.dataset_config['tcga_survival']
        )
        surv_orien = pd.read_csv(
            self.raw_dir / self.dataset_config['orien_survival']
        )
        
        # Harmonize sample IDs
        tcga_expr, orien_expr, surv_tcga, surv_orien = self._harmonize_samples(
            tcga_expr, orien_expr, surv_tcga, surv_orien
        )
        
        # Filter to common genes if configured
        if self.dataset_config.get('use_common_genes', True):
            tcga_expr, orien_expr = self._filter_common_genes(tcga_expr, orien_expr)
        
        # Apply variance filtering if configured
        min_var_percentile = self.dataset_config.get('min_variance_percentile', 0)
        if min_var_percentile > 0:
            tcga_expr, orien_expr = self._filter_low_variance_genes(
                tcga_expr, orien_expr, min_var_percentile
            )
        
        # Standardize if configured
        if self.dataset_config.get('standardize', True):
            tcga_expr = self._standardize(tcga_expr)
            orien_expr = self._standardize(orien_expr)
        
        # Log final statistics
        self._log_dataset_stats(tcga_expr, orien_expr, surv_tcga, surv_orien)
        
        return {
            'tcga_expr': tcga_expr,
            'orien_expr': orien_expr,
            'surv_tcga': surv_tcga,
            'surv_orien': surv_orien
        }
    
    def _harmonize_samples(
        self,
        tcga_expr: pd.DataFrame,
        orien_expr: pd.DataFrame,
        surv_tcga: pd.DataFrame,
        surv_orien: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Harmonize sample IDs between expression and survival data."""
        
        # Get matching samples
        tcga_expr_samples = set(tcga_expr.columns)
        tcga_surv_samples = set(surv_tcga['sampleID'])
        orien_expr_samples = set(orien_expr.columns)
        orien_surv_samples = set(surv_orien['sampleID'])
        
        tcga_matched = sorted(list(tcga_expr_samples.intersection(tcga_surv_samples)))
        orien_matched = sorted(list(orien_expr_samples.intersection(orien_surv_samples)))
        
        logger.info(f"  TCGA: {len(tcga_matched)}/{len(tcga_surv_samples)} samples matched")
        logger.info(f"  ORIEN: {len(orien_matched)}/{len(orien_surv_samples)} samples matched")
        
        # Filter to matched samples
        tcga_expr = tcga_expr[tcga_matched]
        orien_expr = orien_expr[orien_matched]
        
        surv_tcga = surv_tcga[surv_tcga['sampleID'].isin(tcga_matched)]
        surv_orien = surv_orien[surv_orien['sampleID'].isin(orien_matched)]
        
        surv_tcga.set_index('sampleID', inplace=True)
        surv_orien.set_index('sampleID', inplace=True)
        
        return tcga_expr, orien_expr, surv_tcga, surv_orien
    
    def _filter_common_genes(
        self,
        tcga_expr: pd.DataFrame,
        orien_expr: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter to genes present in both cohorts."""
        
        tcga_genes = set(tcga_expr.index)
        orien_genes = set(orien_expr.index)
        common_genes = sorted(list(tcga_genes.intersection(orien_genes)))
        
        logger.info(f"  Common genes: {len(common_genes)}")
        
        return tcga_expr.loc[common_genes], orien_expr.loc[common_genes]
    
    def _filter_low_variance_genes(
        self,
        tcga_expr: pd.DataFrame,
        orien_expr: pd.DataFrame,
        percentile: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Remove low-variance genes (bottom percentile%)."""
        
        # Calculate variance in both cohorts
        tcga_var = tcga_expr.var(axis=1)
        orien_var = orien_expr.var(axis=1)
        
        # Get threshold (take max to be conservative)
        tcga_threshold = np.percentile(tcga_var, percentile)
        orien_threshold = np.percentile(orien_var, percentile)
        threshold = max(tcga_threshold, orien_threshold)
        
        # Keep genes above threshold in both cohorts
        keep_genes = (tcga_var >= threshold) & (orien_var >= threshold)
        
        logger.info(
            f"  Variance filtering (bottom {percentile}%): "
            f"{keep_genes.sum()}/{len(keep_genes)} genes retained"
        )
        
        return tcga_expr.loc[keep_genes], orien_expr.loc[keep_genes]
    
    def _standardize(self, expr_df: pd.DataFrame) -> pd.DataFrame:
        """Z-score standardization (mean=0, std=1) per gene."""
        # Calculate mean and std for each gene (row-wise)
        gene_means = expr_df.mean(axis=1)
        gene_stds = expr_df.std(axis=1)
        
        # Standardize: (x - mean) / std for each gene
        # Using subtract and divide to handle broadcasting properly
        standardized = expr_df.subtract(gene_means, axis=0).divide(gene_stds, axis=0)
        
        return standardized
    
    def _log_dataset_stats(
        self,
        tcga_expr: pd.DataFrame,
        orien_expr: pd.DataFrame,
        surv_tcga: pd.DataFrame,
        surv_orien: pd.DataFrame
    ):
        """Log final dataset statistics."""
        logger.info("="*60)
        logger.info("Dataset Statistics:")
        logger.info(f"  TCGA: {tcga_expr.shape[1]} samples, {tcga_expr.shape[0]} genes")
        logger.info(f"  ORIEN: {orien_expr.shape[1]} samples, {orien_expr.shape[0]} genes")
        logger.info(f"  TCGA events: {surv_tcga['event'].sum()}/{len(surv_tcga)} "
                   f"({100*surv_tcga['event'].mean():.1f}%)")
        logger.info(f"  ORIEN events: {surv_orien['event'].sum()}/{len(surv_orien)} "
                   f"({100*surv_orien['event'].mean():.1f}%)")
        logger.info("="*60)


def load_dataset_from_config(config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load dataset from config.
    
    Args:
        config: Experiment configuration dict
        
    Returns:
        dict with expression and survival dataframes
    """
    factory = DatasetFactory(config)
    return factory.load_dataset()


# Example usage
if __name__ == "__main__":
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # Load a config file
    with open("config/experiments/deepsurv_full.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    data = load_dataset_from_config(config)
    
    print(f"\nLoaded dataset: {config['dataset']['name']}")
    print(f"TCGA shape: {data['tcga_expr'].shape}")
    print(f"ORIEN shape: {data['orien_expr'].shape}")
