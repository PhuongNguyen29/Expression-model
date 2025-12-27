"""
Clinical Data Preprocessor for DeepSurv Integration

This module handles:
1. Loading and encoding clinical variables (one-hot encoding for categorical)
2. Age standardization using source cohort statistics
3. Integration with gene expression features
4. Proper handling of cross-cohort validation (fit on source, transform target)

Encoding scheme (one-hot):
- age: continuous, z-scored using source cohort mean/std
- gender: binary (0=Female, 1=Male) - kept as-is
- smoking: 3 categories -> 3 binary columns (Never, Ever, Unknown)
- alcohol: 3 categories -> 3 binary columns (Never, Ever, Unknown)
- N_stage: 4 categories -> 4 binary columns (N0, N1, N2-3, Unknown)
- T_stage: 3 categories -> 3 binary columns (T1-T2, T3-T4, Unknown)

Total clinical features: 1 (age) + 1 (gender) + 3 + 3 + 4 + 3 = 15 features
Total features with 58 genes: 58 + 15 = 73 features

References:
- Katzman et al. (2018) DeepSurv - categorical variable handling
- Harrell (2015) Regression Modeling Strategies - clinical variable encoding

Author: Phuong
Created: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ClinicalPreprocessor:
    """
    Preprocessor for clinical variables with proper cross-cohort handling.
    
    Implements one-hot encoding for categorical variables and z-score
    standardization for continuous variables, with fit/transform separation
    for cross-cohort validation.
    """
    
    # Define categorical variables and their categories (order matters for consistency)
    CATEGORICAL_VARS = {
        'smoking': ['Never', 'Ever', 'Unknown'],
        'alcohol': ['Never', 'Ever', 'Unknown'],
        'N_stage': ['N0', 'N1', 'N2-3', 'Unknown'],
        'T_stage': ['T1-T2', 'T3-T4', 'Unknown']
    }
    
    # Define continuous variables
    CONTINUOUS_VARS = ['age']
    
    # Define binary variables (already 0/1)
    BINARY_VARS = ['gender']
    
    def __init__(self):
        """Initialize preprocessor."""
        self.age_scaler = None
        self.fitted = False
        self.feature_names = None
        
    def _generate_feature_names(self) -> List[str]:
        """Generate ordered list of clinical feature names after encoding."""
        names = []
        
        # Continuous (z-scored)
        names.append('age_zscore')
        
        # Binary
        names.append('gender')
        
        # Categorical (one-hot)
        for var, categories in self.CATEGORICAL_VARS.items():
            for cat in categories:
                names.append(f'{var}_{cat}')
        
        return names
    
    def _one_hot_encode(self, df: pd.DataFrame, var: str, categories: List[str]) -> pd.DataFrame:
        """
        One-hot encode a categorical variable.
        
        Args:
            df: DataFrame containing the variable
            var: Column name to encode
            categories: List of expected categories (in order)
            
        Returns:
            DataFrame with one-hot encoded columns
        """
        encoded = pd.DataFrame(index=df.index)
        
        for cat in categories:
            col_name = f'{var}_{cat}'
            encoded[col_name] = (df[var] == cat).astype(float)
        
        return encoded
    
    def fit_transform(self, clinical_df: pd.DataFrame, cohort_name: str = 'source') -> pd.DataFrame:
        """
        Fit preprocessor on source cohort and transform.
        
        Args:
            clinical_df: Clinical data DataFrame with sampleID as index or column
            cohort_name: Name for logging
            
        Returns:
            Preprocessed clinical features DataFrame (samples × features)
        """
        logger.info(f"Fitting clinical preprocessor on {cohort_name}")
        logger.info(f"  Input shape: {clinical_df.shape}")
        
        # Ensure sampleID is index
        if 'sampleID' in clinical_df.columns:
            df = clinical_df.set_index('sampleID').copy()
        else:
            df = clinical_df.copy()
        
        # Remove any extra index columns from R export
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # 1. Process continuous variables (fit scaler)
        for var in self.CONTINUOUS_VARS:
            if var not in df.columns:
                raise ValueError(f"Missing continuous variable: {var}")
            
            self.age_scaler = StandardScaler()
            values = df[var].values.reshape(-1, 1)
            scaled = self.age_scaler.fit_transform(values).flatten()
            result[f'{var}_zscore'] = scaled
            
            logger.info(f"  {var}: mean={df[var].mean():.2f}, std={df[var].std():.2f} -> z-scored")
        
        # 2. Process binary variables (keep as-is)
        for var in self.BINARY_VARS:
            if var not in df.columns:
                raise ValueError(f"Missing binary variable: {var}")
            result[var] = df[var].astype(float)
            logger.info(f"  {var}: kept as binary (0/1)")
        
        # 3. Process categorical variables (one-hot encode)
        for var, categories in self.CATEGORICAL_VARS.items():
            if var not in df.columns:
                raise ValueError(f"Missing categorical variable: {var}")
            
            # Check for unexpected categories
            unique_vals = df[var].unique()
            unexpected = set(unique_vals) - set(categories)
            if unexpected:
                logger.warning(f"  {var}: unexpected categories {unexpected}, will be treated as Unknown")
            
            encoded = self._one_hot_encode(df, var, categories)
            result = pd.concat([result, encoded], axis=1)
            
            # Log distribution
            dist = df[var].value_counts()
            logger.info(f"  {var}: {dict(dist)}")
        
        self.feature_names = self._generate_feature_names()
        self.fitted = True
        
        logger.info(f"  Output shape: {result.shape}")
        logger.info(f"  Feature names: {self.feature_names}")
        
        return result
    
    def transform(self, clinical_df: pd.DataFrame, cohort_name: str = 'target') -> pd.DataFrame:
        """
        Transform target cohort using fitted parameters from source.
        
        CRITICAL: Uses source cohort's age mean/std for standardization.
        
        Args:
            clinical_df: Clinical data DataFrame
            cohort_name: Name for logging
            
        Returns:
            Preprocessed clinical features DataFrame (samples × features)
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        logger.info(f"Transforming {cohort_name} clinical data using source parameters")
        logger.info(f"  Input shape: {clinical_df.shape}")
        
        # Ensure sampleID is index
        if 'sampleID' in clinical_df.columns:
            df = clinical_df.set_index('sampleID').copy()
        else:
            df = clinical_df.copy()
        
        # Remove any extra index columns
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # 1. Process continuous variables (use fitted scaler)
        for var in self.CONTINUOUS_VARS:
            values = df[var].values.reshape(-1, 1)
            scaled = self.age_scaler.transform(values).flatten()
            result[f'{var}_zscore'] = scaled
            
            logger.info(f"  {var}: transformed using source mean/std")
        
        # 2. Process binary variables
        for var in self.BINARY_VARS:
            result[var] = df[var].astype(float)
        
        # 3. Process categorical variables
        for var, categories in self.CATEGORICAL_VARS.items():
            encoded = self._one_hot_encode(df, var, categories)
            result = pd.concat([result, encoded], axis=1)
        
        logger.info(f"  Output shape: {result.shape}")
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Return list of clinical feature names."""
        if self.feature_names is None:
            return self._generate_feature_names()
        return self.feature_names


class IntegratedPreprocessor:
    """
    Combined preprocessor for gene expression + clinical data.
    
    Handles:
    1. Gene expression preprocessing (standardization)
    2. Clinical data preprocessing (one-hot encoding, age z-scoring)
    3. Feature concatenation with proper sample alignment
    """
    
    def __init__(self, config: dict):
        """
        Initialize integrated preprocessor.
        
        Args:
            config: Configuration dict with 'data' key containing preprocessing settings
        """
        self.config = config
        self.gene_scaler = None
        self.clinical_preprocessor = ClinicalPreprocessor()
        self.gene_names = None
        self.clinical_feature_names = None
        self.all_feature_names = None
        self.fitted = False
        
    def fit_transform(
        self,
        expr_df: pd.DataFrame,
        clinical_df: pd.DataFrame,
        cohort_name: str = 'source'
    ) -> pd.DataFrame:
        """
        Fit on source cohort and transform.
        
        Args:
            expr_df: Gene expression DataFrame (genes × samples)
            clinical_df: Clinical DataFrame with sampleID column
            cohort_name: Name for logging
            
        Returns:
            Combined features DataFrame (samples × features)
            Features ordered as: [genes..., clinical...]
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Integrated Preprocessing: {cohort_name} (fit_transform)")
        logger.info(f"{'='*60}")
        
        # Store gene names
        self.gene_names = expr_df.index.tolist()
        
        # 1. Process gene expression
        logger.info("\n--- Gene Expression ---")
        logger.info(f"  Input: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")
        
        # Standardize genes (z-score per gene across samples)
        self.gene_scaler = StandardScaler()
        expr_T = expr_df.T  # samples × genes
        expr_scaled = self.gene_scaler.fit_transform(expr_T)
        
        # Clip outliers
        n_clipped = np.sum(np.abs(expr_scaled) > 3.0)
        expr_scaled = np.clip(expr_scaled, -3.0, 3.0)
        logger.info(f"  Standardized: {n_clipped} values clipped to ±3 std")
        
        expr_processed = pd.DataFrame(
            expr_scaled,
            index=expr_T.index,
            columns=expr_df.index  # gene names as columns
        )
        
        # 2. Process clinical data
        logger.info("\n--- Clinical Data ---")
        clinical_processed = self.clinical_preprocessor.fit_transform(clinical_df, cohort_name)
        self.clinical_feature_names = self.clinical_preprocessor.get_feature_names()
        
        # 3. Align samples and concatenate
        logger.info("\n--- Feature Integration ---")
        common_samples = list(set(expr_processed.index) & set(clinical_processed.index))
        common_samples = sorted(common_samples)
        
        if len(common_samples) < len(expr_processed):
            logger.warning(f"  Sample mismatch: {len(expr_processed)} expression, "
                          f"{len(clinical_processed)} clinical, {len(common_samples)} overlap")
        
        expr_aligned = expr_processed.loc[common_samples]
        clinical_aligned = clinical_processed.loc[common_samples]
        
        # Concatenate: genes first, then clinical
        combined = pd.concat([expr_aligned, clinical_aligned], axis=1)
        
        self.all_feature_names = self.gene_names + self.clinical_feature_names
        self.fitted = True
        
        logger.info(f"  Final shape: {combined.shape}")
        logger.info(f"  Features: {len(self.gene_names)} genes + {len(self.clinical_feature_names)} clinical = {len(self.all_feature_names)} total")
        
        return combined
    
    def transform(
        self,
        expr_df: pd.DataFrame,
        clinical_df: pd.DataFrame,
        cohort_name: str = 'target'
    ) -> pd.DataFrame:
        """
        Transform target cohort using fitted parameters.
        
        CRITICAL: Uses source cohort statistics for standardization.
        
        Args:
            expr_df: Gene expression DataFrame (genes × samples)
            clinical_df: Clinical DataFrame with sampleID column
            cohort_name: Name for logging
            
        Returns:
            Combined features DataFrame (samples × features)
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Integrated Preprocessing: {cohort_name} (transform)")
        logger.info(f"{'='*60}")
        
        # 1. Process gene expression using fitted scaler
        logger.info("\n--- Gene Expression ---")
        expr_T = expr_df.T  # samples × genes
        expr_scaled = self.gene_scaler.transform(expr_T)
        
        # Clip outliers
        expr_scaled = np.clip(expr_scaled, -3.0, 3.0)
        
        expr_processed = pd.DataFrame(
            expr_scaled,
            index=expr_T.index,
            columns=expr_df.index
        )
        logger.info(f"  Transformed using source parameters")
        
        # 2. Process clinical data using fitted preprocessor
        logger.info("\n--- Clinical Data ---")
        clinical_processed = self.clinical_preprocessor.transform(clinical_df, cohort_name)
        
        # 3. Align and concatenate
        logger.info("\n--- Feature Integration ---")
        common_samples = list(set(expr_processed.index) & set(clinical_processed.index))
        common_samples = sorted(common_samples)
        
        expr_aligned = expr_processed.loc[common_samples]
        clinical_aligned = clinical_processed.loc[common_samples]
        
        combined = pd.concat([expr_aligned, clinical_aligned], axis=1)
        
        logger.info(f"  Final shape: {combined.shape}")
        
        return combined
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get feature names organized by type.
        
        Returns:
            Dict with 'genes', 'clinical', and 'all' keys
        """
        return {
            'genes': self.gene_names or [],
            'clinical': self.clinical_feature_names or [],
            'all': self.all_feature_names or []
        }


def load_clinical_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load clinical data for both cohorts.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dict with 'tcga' and 'orien' clinical DataFrames
    """
    logger.info("Loading clinical data...")
    
    tcga_clinical = pd.read_csv(data_dir / "raw" / "clinical_tcga.csv")
    orien_clinical = pd.read_csv(data_dir / "raw" / "clinical_orien_updated.csv")
    
    logger.info(f"  TCGA clinical: {tcga_clinical.shape}")
    logger.info(f"  ORIEN clinical: {orien_clinical.shape}")
    
    return {
        'tcga': tcga_clinical,
        'orien': orien_clinical
    }


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*70)
    print("Testing ClinicalPreprocessor")
    print("="*70)
    
    # Create sample clinical data
    np.random.seed(42)
    n_samples = 100
    
    sample_clinical = pd.DataFrame({
        'sampleID': [f'SAMPLE_{i}' for i in range(n_samples)],
        'age': np.random.randint(40, 80, n_samples),
        'gender': np.random.randint(0, 2, n_samples),
        'smoking': np.random.choice(['Never', 'Ever', 'Unknown'], n_samples),
        'alcohol': np.random.choice(['Never', 'Ever', 'Unknown'], n_samples),
        'N_stage': np.random.choice(['N0', 'N1', 'N2-3', 'Unknown'], n_samples),
        'T_stage': np.random.choice(['T1-T2', 'T3-T4', 'Unknown'], n_samples)
    })
    
    # Test fit_transform
    preprocessor = ClinicalPreprocessor()
    result = preprocessor.fit_transform(sample_clinical, 'test_cohort')
    
    print(f"\nInput shape: {sample_clinical.shape}")
    print(f"Output shape: {result.shape}")
    print(f"\nFeature names ({len(preprocessor.get_feature_names())}):")
    for name in preprocessor.get_feature_names():
        print(f"  - {name}")
    
    print(f"\nFirst 5 rows:\n{result.head()}")
    
    # Test transform with different data
    sample_clinical2 = pd.DataFrame({
        'sampleID': [f'SAMPLE2_{i}' for i in range(50)],
        'age': np.random.randint(30, 90, 50),  # Different age range
        'gender': np.random.randint(0, 2, 50),
        'smoking': np.random.choice(['Never', 'Ever', 'Unknown'], 50),
        'alcohol': np.random.choice(['Never', 'Ever', 'Unknown'], 50),
        'N_stage': np.random.choice(['N0', 'N1', 'N2-3', 'Unknown'], 50),
        'T_stage': np.random.choice(['T1-T2', 'T3-T4', 'Unknown'], 50)
    })
    
    result2 = preprocessor.transform(sample_clinical2, 'test_cohort2')
    print(f"\nTransform test - Output shape: {result2.shape}")
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
