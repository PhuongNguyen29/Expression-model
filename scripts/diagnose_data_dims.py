"""
Diagnostic script to check data dimensions at each step
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import torch
from src.data.dataset import SurvivalDataset

print("="*60)
print("DATA DIMENSION DIAGNOSTIC")
print("="*60)

# Step 1: Check raw preprocessed data files
print("\n1. CHECKING PREPROCESSED CSV FILES:")
print("-"*40)

tcga_expr = pd.read_csv("data/processed/tcga_preprocessed.csv", index_col=0)
orien_expr = pd.read_csv("data/processed/orien_preprocessed.csv", index_col=0)
surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)

print(f"TCGA expression shape: {tcga_expr.shape}")
print(f"  → Rows (index): {tcga_expr.index.name or 'unnamed'} - First 3: {list(tcga_expr.index[:3])}")
print(f"  → Columns: First 3: {list(tcga_expr.columns[:3])}")
print(f"  → Data orientation: {'Genes × Samples' if tcga_expr.shape[0] > tcga_expr.shape[1] else 'Samples × Genes'}")

print(f"\nORIEN expression shape: {orien_expr.shape}")
print(f"  → Rows (index): {orien_expr.index.name or 'unnamed'} - First 3: {list(orien_expr.index[:3])}")
print(f"  → Columns: First 3: {list(orien_expr.columns[:3])}")
print(f"  → Data orientation: {'Genes × Samples' if orien_expr.shape[0] > orien_expr.shape[1] else 'Samples × Genes'}")

print(f"\nSurvival data shapes:")
print(f"  → TCGA survival: {surv_tcga.shape}")
print(f"  → ORIEN survival: {surv_orien.shape}")

# Step 2: Check what SurvivalDataset does
print("\n2. CHECKING SURVIVALDATASET CLASS:")
print("-"*40)

# Create dataset
dataset = SurvivalDataset(tcga_expr, surv_tcga)

print(f"Dataset length: {len(dataset)}")
print(f"Dataset n_features attribute: {dataset.n_features}")

# Get a sample
sample = dataset[0]
print(f"\nFirst sample from dataset:")
print(f"  → Features shape: {sample['features'].shape}")
print(f"  → Features type: {type(sample['features'])}")
print(f"  → Time: {sample['time']:.2f}")
print(f"  → Event: {sample['event']}")

# Check the internal X matrix
print(f"\nInternal dataset.X shape: {dataset.X.shape}")
print(f"  → This should be (n_samples, n_features) = (339, 14778)")

# Step 3: Check DataLoader output
print("\n3. CHECKING DATALOADER OUTPUT:")
print("-"*40)

from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=False)
batch = next(iter(loader))

print(f"Batch features shape: {batch['features'].shape}")
print(f"  → Expected: (batch_size=32, n_features=14778)")
print(f"Batch time shape: {batch['time'].shape}")
print(f"Batch event shape: {batch['event'].shape}")

# Step 4: Identify the problem
print("\n4. PROBLEM IDENTIFICATION:")
print("-"*40)

if tcga_expr.shape[0] > tcga_expr.shape[1]:
    print("✓ Data is stored as Genes × Samples (14778 × 339)")
    print("  → tcga_expr.shape[0] = number of genes = 14778")
    print("  → tcga_expr.shape[1] = number of samples = 339")
    correct_n_features = tcga_expr.shape[0]
else:
    print("✓ Data is stored as Samples × Genes (339 × 14778)")
    print("  → tcga_expr.shape[0] = number of samples = 339")
    print("  → tcga_expr.shape[1] = number of genes = 14778")
    correct_n_features = tcga_expr.shape[1]

print(f"\n✅ CORRECT n_features should be: {correct_n_features}")

# Step 5: Show the fix needed in train_deepsurv.py
print("\n5. FIX NEEDED IN train_deepsurv.py:")
print("-"*40)
print("Look for this line in run_bidirectional_experiments():")
print("  n_features = tcga_expr.shape[1]  # WRONG if data is genes × samples")
print("\nChange it to:")
if tcga_expr.shape[0] > tcga_expr.shape[1]:
    print("  n_features = tcga_expr.shape[0]  # Number of genes (rows)")
    print("\nOR more robustly:")
    print("  # Determine correct dimension based on data orientation")
    print("  if tcga_expr.shape[0] > tcga_expr.shape[1]:  # genes × samples")
    print("      n_features = tcga_expr.shape[0]")
    print("  else:  # samples × genes")
    print("      n_features = tcga_expr.shape[1]")
else:
    print("  n_features = tcga_expr.shape[1]  # This is already correct")

print("\n" + "="*60)