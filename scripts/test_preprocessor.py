import sys
sys.path.append('.')

from src.data.data_loader import DataLoader
from src.data.preprocessor import GeneExpressionPreprocessor
import yaml
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_preprocessor():
    """Test the preprocessing pipeline"""
    print("="*60)
    print("TESTING PREPROCESSOR")
    print("="*60)
    
    # Load config
    with open("config/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Load data
    logger.info("Loading data...")
    loader = DataLoader()
    data = loader.run_full_pipeline()
    
    # Step 2: Initialize preprocessor
    preprocessor = GeneExpressionPreprocessor(config)
    
    # Step 3: Fit and transform
    logger.info("\n" + "="*60)
    logger.info("Testing preprocessing...")
    processed = preprocessor.fit_transform(
        data['tcga_expr'],
        data['orien_expr']
    )
    
    # Step 4: Verify results
    print("\n" + "="*60)
    print("VERIFICATION CHECKS")
    print("="*60)
    
    # Check shapes
    tcga_proc = processed['tcga_processed']
    orien_proc = processed['orien_processed']
    
    print(f"✓ Gene count consistency: {tcga_proc.shape[0] == orien_proc.shape[0]}")
    print(f"✓ Genes after filtering: {tcga_proc.shape[0]}")
    print(f"✓ Reduction from original: {data['tcga_expr'].shape[0]} → {tcga_proc.shape[0]}")
    print(f"✓ Percentage retained: {100*tcga_proc.shape[0]/data['tcga_expr'].shape[0]:.1f}%")
    
    # Check standardization
    if config['data']['standardize']:
        tcga_mean = tcga_proc.values.mean()
        tcga_std = tcga_proc.values.std()
        orien_mean = orien_proc.values.mean()
        orien_std = orien_proc.values.std()
        
        print(f"\n✓ TCGA standardized - Mean: {tcga_mean:.4f} (should be ~0)")
        print(f"✓ TCGA standardized - Std: {tcga_std:.4f} (should be ~1)")
        print(f"✓ ORIEN standardized - Mean: {orien_mean:.4f} (should be ~0)")
        print(f"✓ ORIEN standardized - Std: {orien_std:.4f} (should be ~1)")
    
    # Check gene variance distribution
    gene_vars = processed['gene_variances']
    print(f"\n✓ Gene variance range: [{gene_vars.min():.4f}, {gene_vars.max():.4f}]")
    print(f"✓ Median gene variance: {gene_vars.median():.4f}")
    
    # Check that same genes are in both cohorts
    assert list(tcga_proc.index) == list(orien_proc.index), "Gene mismatch between cohorts!"
    print(f"\n✓ Same genes in both cohorts: TRUE")
    
    # Save preprocessed data
    print("\n" + "="*60)
    print("Saving preprocessed data...")
    tcga_proc.to_csv("data/processed/tcga_preprocessed.csv")
    orien_proc.to_csv("data/processed/orien_preprocessed.csv")
    preprocessor.save("data/processed/preprocessor.pkl")
    
    print("✓ Data saved to data/processed/")
    
    # Summary statistics
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Initial common genes: {data['tcga_expr'].shape[0]}")
    print(f"After variance filter: {tcga_proc.shape[0]}")
    print(f"Genes removed: {data['tcga_expr'].shape[0] - tcga_proc.shape[0]}")
    print(f"TCGA samples: {tcga_proc.shape[1]}")
    print(f"ORIEN samples: {orien_proc.shape[1]}")
    
    print("\n" + "="*60)
    print("PREPROCESSOR TEST PASSED!")
    print("="*60)
    
    return processed

if __name__ == "__main__":
    processed = test_preprocessor()