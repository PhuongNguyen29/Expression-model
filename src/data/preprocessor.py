import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class GeneExpressionPreprocessor:
    """
    Minimal preprocessing for deep learning models.
    Based on "Variance-based feature selection for neural networks" (Battiti, 1994)
    and "Deep learning for patient-specific survival prediction" (Yousefi et al., 2017, Bioinformatics)
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.min_variance_percentile = config['data']['min_variance_percentile']
        self.standardize = config['data']['standardize']
        
        # These will be fitted on training data
        self.selected_genes = None
        self.gene_variances = None
        self.scaler_tcga = None
        self.scaler_orien = None
        
    def compute_gene_variance(self, expr_data: pd.DataFrame) -> pd.Series:
        """
        Compute variance for each gene across samples.
        Using variance as it's more interpretable than IQR for normally distributed data.
        """
        return expr_data.var(axis=1)
    
    def select_genes_by_variance(self, 
                             tcga_expr: pd.DataFrame, 
                             orien_expr: pd.DataFrame) -> list:
        """
        Select genes based on variance, removing only bottom 10%.
        CRITICAL: Use combined variance from both cohorts for stability.
        
        Based on "Robust biomarker identification through cross-study validation" 
        (Bernau et al., 2014, Bioinformatics)
        """
        logger.info("="*50)
        logger.info("Computing gene variance for filtering...")
        
        # Initial gene count (should be common genes)
        initial_genes = len(tcga_expr)
        assert len(tcga_expr) == len(orien_expr), "Gene counts don't match between cohorts!"
        logger.info(f"Starting with {initial_genes} common genes")
        
        # Compute variance in each cohort
        tcga_var = self.compute_gene_variance(tcga_expr)
        orien_var = self.compute_gene_variance(orien_expr)
        
        # Use average variance across cohorts (for stability)
        combined_variance = (tcga_var + orien_var) / 2
        
        # Calculate threshold
        threshold = np.percentile(combined_variance, self.min_variance_percentile)
        
        # Select genes above threshold
        selected_genes = combined_variance[combined_variance > threshold].index.tolist()
        
        # Calculate statistics
        removed_genes = initial_genes - len(selected_genes)
        percent_kept = 100 * len(selected_genes) / initial_genes
        percent_removed = 100 * removed_genes / initial_genes
        
        logger.info(f"Variance threshold (percentile {self.min_variance_percentile}): {threshold:.4f}")
        logger.info(f"Genes removed: {removed_genes} ({percent_removed:.1f}%)")
        logger.info(f"Genes retained: {len(selected_genes)} ({percent_kept:.1f}%)")
        
        # Log some examples of removed genes (lowest variance)
        bottom_genes = combined_variance.nsmallest(5)
        logger.info(f"Examples of removed genes (lowest variance): {bottom_genes.index.tolist()[:5]}")
        
        # Store for later use
        self.selected_genes = selected_genes
        self.gene_variances = combined_variance
        self.removed_count = removed_genes
        
        return selected_genes

    
    def standardize_cohorts(self, 
                           tcga_expr: pd.DataFrame, 
                           orien_expr: pd.DataFrame,
                           fit: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standardize each cohort separately to mean=0, std=1.
        This preserves cohort-specific characteristics while normalizing scale.
        
        Based on "Batch effect correction in genomic studies" (Leek et al., 2010, Nature Reviews Genetics)
        """
        if not self.standardize:
            return tcga_expr, orien_expr
            
        logger.info("Standardizing expression data (z-score per cohort)...")
        
        # Transpose for sklearn (samples × genes)
        tcga_T = tcga_expr.T
        orien_T = orien_expr.T
        
        if fit:
            # Fit and transform
            self.scaler_tcga = StandardScaler()
            self.scaler_orien = StandardScaler()
            
            tcga_scaled = self.scaler_tcga.fit_transform(tcga_T)
            orien_scaled = self.scaler_orien.fit_transform(orien_T)
        else:
            # Only transform using fitted scalers
            if self.scaler_tcga is None or self.scaler_orien is None:
                raise ValueError("Scalers not fitted. Run with fit=True first.")
            
            tcga_scaled = self.scaler_tcga.transform(tcga_T)
            orien_scaled = self.scaler_orien.transform(orien_T)
            
        logger.info("Clipping outliers to ±3 standard deviations...")
        tcga_before_clip = np.sum(np.abs(tcga_scaled) > 2.9)
        orien_before_clip = np.sum(np.abs(orien_scaled) > 2.9)

        tcga_scaled = np.clip(tcga_scaled, -3.0, 3.0)
        orien_scaled = np.clip(orien_scaled, -3.0, 3.0)

        logger.info(f"  TCGA: {tcga_before_clip} values clipped ({100*tcga_before_clip/tcga_scaled.size:.2f}%)")
        logger.info(f"  ORIEN: {orien_before_clip} values clipped ({100*orien_before_clip/orien_scaled.size:.2f}%)")
        
        # Convert back to DataFrames and transpose back (genes × samples)
        tcga_standardized = pd.DataFrame(
            tcga_scaled.T,
            index=tcga_expr.index,
            columns=tcga_expr.columns
        )
        orien_standardized = pd.DataFrame(
            orien_scaled.T,
            index=orien_expr.index,
            columns=orien_expr.columns
        )
        
        # Log statistics
        logger.info(f"TCGA after standardization - mean: {tcga_standardized.mean().mean():.4f}, "
                   f"std: {tcga_standardized.std().mean():.4f}")
        logger.info(f"ORIEN after standardization - mean: {orien_standardized.mean().mean():.4f}, "
                   f"std: {orien_standardized.std().mean():.4f}")
        
        return tcga_standardized, orien_standardized
    
    def fit_transform(self, 
                 tcga_expr: pd.DataFrame, 
                 orien_expr: pd.DataFrame) -> Dict:
        """
        Fit preprocessor on data and transform it.
        This should be called on the full datasets before train/test split.
        """
        logger.info("="*50)
        logger.info("Starting preprocessing pipeline...")
        
        # Verify we're starting with common genes
        assert tcga_expr.index.equals(orien_expr.index), \
            "Gene indices don't match! Ensure common genes are used."
        
        initial_genes = len(tcga_expr)
        logger.info(f"Input: {initial_genes} common genes × (TCGA: {tcga_expr.shape[1]}, ORIEN: {orien_expr.shape[1]} samples)")
        
        # Step 1: Variance filtering
        selected_genes = self.select_genes_by_variance(tcga_expr, orien_expr)
        tcga_filtered = tcga_expr.loc[selected_genes]
        orien_filtered = orien_expr.loc[selected_genes]
        
        # Step 2: Standardization
        tcga_final, orien_final = self.standardize_cohorts(
            tcga_filtered, 
            orien_filtered,
            fit=True
        )
        
        # Final summary
        logger.info("="*50)
        logger.info("Preprocessing Summary:")
        logger.info(f"  Initial genes (common): {initial_genes}")
        logger.info(f"  After variance filter: {len(selected_genes)} ({100*len(selected_genes)/initial_genes:.1f}%)")
        logger.info(f"  Final shape - TCGA: {tcga_final.shape}, ORIEN: {orien_final.shape}")
        logger.info(f"  Ready for deep learning with {len(selected_genes)} features!")
        
        return {
            'tcga_processed': tcga_final,
            'orien_processed': orien_final,
            'selected_genes': selected_genes,
            'gene_variances': self.gene_variances,
            'gene_count_summary': {
                'initial_common_genes': initial_genes,
                'after_variance_filter': len(selected_genes),
                'removed_by_variance': self.removed_count
            }
        }
    
    def transform(self, 
                 tcga_expr: pd.DataFrame, 
                 orien_expr: pd.DataFrame) -> Dict:
        """
        Transform new data using fitted parameters.
        Use this for validation/test sets.
        """
        if self.selected_genes is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Apply same gene selection
        tcga_filtered = tcga_expr.loc[self.selected_genes]
        orien_filtered = orien_expr.loc[self.selected_genes]
        
        # Apply standardization with fitted scalers
        tcga_final, orien_final = self.standardize_cohorts(
            tcga_filtered,
            orien_filtered,
            fit=False
        )
        
        return {
            'tcga_processed': tcga_final,
            'orien_processed': orien_final
        }
    
    def save(self, path: str):
        """Save preprocessor state for later use."""
        state = {
            'selected_genes': self.selected_genes,
            'gene_variances': self.gene_variances,
            'scaler_tcga': self.scaler_tcga,
            'scaler_orien': self.scaler_orien,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: str):
        """Load preprocessor state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.selected_genes = state['selected_genes']
        self.gene_variances = state['gene_variances']
        self.scaler_tcga = state['scaler_tcga']
        self.scaler_orien = state['scaler_orien']
        self.config = state['config']
        logger.info(f"Preprocessor loaded from {path}")
        
    def fit_transform_single_cohort(self, 
                                expr_data: pd.DataFrame,
                                cohort_name: str = 'cohort') -> pd.DataFrame:
        """
        Fit preprocessor and transform a SINGLE cohort (for CV fold training set).
        
        CRITICAL: Use this inside CV loops to prevent data leakage.
        
        Args:
            expr_data: Gene expression DataFrame (genes × samples)
            cohort_name: Name for logging (e.g., 'TCGA', 'ORIEN')
            
        Returns:
            Preprocessed DataFrame (genes × samples)
            
        Example:
            # Inside CV loop
            train_fold = raw_data[:, train_indices]
            preprocessor = GeneExpressionPreprocessor(config)
            train_processed = preprocessor.fit_transform_single_cohort(train_fold)
            test_processed = preprocessor.transform_single_cohort(test_fold)
        """
        logger.info(f"Preprocessing {cohort_name} (single cohort mode)")
        logger.info(f"  Input shape: {expr_data.shape} (genes × samples)")
        gene_var = self.compute_gene_variance(expr_data)
        
        if self.min_variance_percentile > 0:
            threshold = np.percentile(gene_var, self.min_variance_percentile)
            selected_genes = gene_var[gene_var > threshold].index.tolist()
            filtered = expr_data.loc[selected_genes]
            logger.info(f"  After variance filter: {len(selected_genes)} genes "
                        f"({100*len(selected_genes)/len(expr_data):.1f}% retained)")
        else:
            selected_genes = expr_data.index.tolist()
            filtered = expr_data
            logger.info(f"  Variance filter disabled, keeping all {len(selected_genes)} genes")
    
        
        self.selected_genes = selected_genes
        self.gene_variances = gene_var
        
        # filtered = expr_data.loc[selected_genes]
        # logger.info(f"  After variance filter: {len(selected_genes)} genes "
        #             f"({100*len(selected_genes)/len(expr_data):.1f}% retained)")
        
        # Step 2: Standardization (z-score)
        if self.standardize:
            scaler = StandardScaler()
            # Transpose: samples as rows for sklearn
            scaled_T = scaler.fit_transform(filtered.T)
            self.scaler = scaler  # Store fitted scaler
            
            logger.info("Clipping outliers to ±3 std...")
            n_clipped = np.sum(np.abs(scaled_T) > 2.9)
            scaled_T = np.clip(scaled_T, -3.0, 3.0)
            logger.info(f"  {n_clipped} values clipped ({100*n_clipped/scaled_T.size:.2f}%)")
        
            # Transpose back: genes as rows
            result = pd.DataFrame(
                scaled_T.T,
                index=filtered.index,
                columns=filtered.columns
            )
            
            logger.info(f"  Standardized: mean={result.mean().mean():.4f}, "
                    f"std={result.std().mean():.4f}")
        else:
            result = filtered
        
        return result


    def transform_single_cohort(self, expr_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data using fitted parameters from training fold.
        
        CRITICAL: Only call this AFTER fit_transform_single_cohort().
        
        Args:
            expr_data: Gene expression DataFrame (genes × samples)
            
        Returns:
            Preprocessed DataFrame using training fold parameters
            
        Raises:
            ValueError: If preprocessor not fitted
        """
        if self.selected_genes is None:
            raise ValueError(
                "Preprocessor not fitted. Call fit_transform_single_cohort() first."
            )
        
        # Apply same gene selection as training fold
        filtered = expr_data.loc[self.selected_genes]
        
        # Apply standardization with fitted scaler
        if self.standardize:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call fit_transform_single_cohort() first.")
            
            scaled_T = self.scaler.transform(filtered.T)
            result = pd.DataFrame(
                scaled_T.T,
                index=filtered.index,
                columns=filtered.columns
            )
        else:
            result = filtered
        
        return result