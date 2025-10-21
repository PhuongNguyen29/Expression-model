"""
Check how SurvivalDataset handles the data transformation
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np

# Load data
tcga_expr = pd.read_csv("data/processed/tcga_preprocessed.csv", index_col=0)
surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)

print("Original data shape:", tcga_expr.shape)
print("First 3 row indices:", tcga_expr.index[:3].tolist())
print("First 3 column names:", tcga_expr.columns[:3].tolist())

# Check if row indices look like gene names and columns look like sample IDs
first_row_idx = str(tcga_expr.index[0])
first_col_name = str(tcga_expr.columns[0])

print(f"\nFirst row index: {first_row_idx}")
print(f"First column name: {first_col_name}")

if "TCGA" in first_col_name or len(first_col_name) > 20:
    print("\n✓ Columns appear to be sample IDs (contain TCGA or are long IDs)")
    print("✓ Data is likely Genes (rows) × Samples (columns)")
    print("\nYour SurvivalDataset should do:")
    print("  X = expression_df.T.values  # Transpose to get samples × genes")
    print("  n_features = expression_df.shape[0]  # Number of genes (rows)")
elif "TCGA" in first_row_idx or len(first_row_idx) > 20:
    print("\n✓ Rows appear to be sample IDs")
    print("✓ Data is likely Samples (rows) × Genes (columns)")
    print("\nYour SurvivalDataset should do:")
    print("  X = expression_df.values  # Already samples × genes")
    print("  n_features = expression_df.shape[1]  # Number of genes (columns)")

# Now let's see what YOUR SurvivalDataset actually does
print("\n" + "="*60)
print("CHECKING YOUR SURVIVALDATASET CLASS:")
print("="*60)

try:
    from src.data.dataset import SurvivalDataset
    
    # Create dataset
    dataset = SurvivalDataset(tcga_expr, surv_tcga)
    
    print(f"Dataset internal X shape: {dataset.X.shape}")
    print(f"Dataset n_features: {dataset.n_features}")
    print(f"Dataset length: {len(dataset)}")
    
    if dataset.X.shape == (339, 14778):
        print("\n✅ CORRECT: Dataset X is (339 samples, 14778 genes)")
        print("The SurvivalDataset is correctly transposing the data!")
    elif dataset.X.shape == (14778, 339):
        print("\n❌ PROBLEM: Dataset X is (14778, 339) - wrong orientation!")
        print("The SurvivalDataset is NOT transposing correctly")
    
except Exception as e:
    print(f"Error loading SurvivalDataset: {e}")
    print("\nPlease check your src/data/dataset.py file")