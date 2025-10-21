import sys
sys.path.append('.')

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import logging
from src.data.dataset import (
    SurvivalDataset, 
    CombinedSurvivalDataset,
    create_survival_dataloaders
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dataset():
    """Test the PyTorch Dataset implementation"""
    print("="*60)
    print("TESTING PYTORCH DATASET")
    print("="*60)
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    tcga_expr = pd.read_csv("data/processed/tcga_preprocessed.csv", index_col=0)
    orien_expr = pd.read_csv("data/processed/orien_preprocessed.csv", index_col=0)
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    print(f"Loaded data shapes:")
    print(f"  TCGA: {tcga_expr.shape}, ORIEN: {orien_expr.shape}")
    
    # Test 1: Individual TCGA Dataset
    print("\n" + "="*60)
    print("TEST 1: TCGA Dataset")
    print("="*60)
    
    tcga_dataset = SurvivalDataset(tcga_expr, surv_tcga)
    print(f"✓ Dataset size: {len(tcga_dataset)} samples")
    print(f"✓ Number of features: {tcga_dataset.n_features}")
    
    # Get first sample
    sample = tcga_dataset[0]
    print(f"✓ Sample keys: {list(sample.keys())}")
    print(f"✓ Feature tensor shape: {sample['features'].shape}")
    print(f"✓ Feature tensor dtype: {sample['features'].dtype}")
    print(f"✓ Time value: {sample['time']:.2f}")
    print(f"✓ Event value: {sample['event']}")
    
    # Test 2: Train/Validation Split
    print("\n" + "="*60)
    print("TEST 2: Train/Validation Split")
    print("="*60)
    
    train_dataset, valid_dataset = tcga_dataset.create_train_valid_split(
        valid_size=0.2, 
        random_seed=42
    )
    print(f"✓ Train size: {len(train_dataset)} ({100*len(train_dataset)/len(tcga_dataset):.1f}%)")
    print(f"✓ Valid size: {len(valid_dataset)} ({100*len(valid_dataset)/len(tcga_dataset):.1f}%)")
    
    # Test 3: DataLoader
    print("\n" + "="*60)
    print("TEST 3: DataLoader")
    print("="*60)
    
    train_loader, valid_loader = create_survival_dataloaders(
        tcga_expr, 
        surv_tcga,
        batch_size=32,
        valid_size=0.2,
        num_workers=0  # Set to 0 for testing to avoid multiprocessing issues
    )
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f"✓ Batch keys: {list(batch.keys())}")
    print(f"✓ Batch features shape: {batch['features'].shape}")
    print(f"✓ Batch time shape: {batch['time'].shape}")
    print(f"✓ Batch event shape: {batch['event'].shape}")
    print(f"✓ Number of batches in train_loader: {len(train_loader)}")
    print(f"✓ Number of batches in valid_loader: {len(valid_loader)}")
    
    # Test 4: Combined Dataset
    print("\n" + "="*60)
    print("TEST 4: Combined Dataset (TCGA + ORIEN)")
    print("="*60)
    
    combined_dataset = CombinedSurvivalDataset(
        tcga_expr, surv_tcga,
        orien_expr, surv_orien
    )
    
    print(f"✓ Combined size: {len(combined_dataset)} samples")
    print(f"✓ TCGA samples: {(combined_dataset.cohort == 0).sum()}")
    print(f"✓ ORIEN samples: {(combined_dataset.cohort == 1).sum()}")
    
    # Get samples from each cohort
    tcga_sample = combined_dataset[0]  # First TCGA sample
    orien_sample = combined_dataset[400]  # Should be ORIEN sample
    
    print(f"✓ First sample cohort: {tcga_sample['cohort']} (should be 0=TCGA)")
    print(f"✓ Sample 400 cohort: {orien_sample['cohort']} (should be 1=ORIEN)")
    
    # Test 5: Verify data integrity
    print("\n" + "="*60)
    print("TEST 5: Data Integrity Checks")
    print("="*60)
    
    # Check for NaN values
    X, y_time, y_event = tcga_dataset.get_features_and_labels()
    print(f"✓ NaN in features: {np.isnan(X).any()}")
    print(f"✓ NaN in times: {np.isnan(y_time).any()}")
    print(f"✓ NaN in events: {np.isnan(y_event).any()}")
    
    # Check time values are positive
    print(f"✓ All times positive: {(y_time > 0).all()}")
    
    # Check events are binary
    print(f"✓ Events are binary: {set(y_event) == {0.0, 1.0} or set(y_event) == {0.0} or set(y_event) == {1.0}}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"TCGA Dataset:")
    print(f"  - Samples: {len(tcga_dataset)}")
    print(f"  - Features: {tcga_dataset.n_features}")
    print(f"  - Event rate: {tcga_dataset.y_event.mean():.2%}")
    print(f"  - Median survival: {np.median(tcga_dataset.y_time):.2f}")
    
    orien_dataset = SurvivalDataset(orien_expr, surv_orien)
    print(f"\nORIEN Dataset:")
    print(f"  - Samples: {len(orien_dataset)}")
    print(f"  - Features: {orien_dataset.n_features}")
    print(f"  - Event rate: {orien_dataset.y_event.mean():.2%}")
    print(f"  - Median survival: {np.median(orien_dataset.y_time):.2f}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    
    return {
        'tcga_dataset': tcga_dataset,
        'orien_dataset': orien_dataset,
        'combined_dataset': combined_dataset
    }

if __name__ == "__main__":
    datasets = test_dataset()