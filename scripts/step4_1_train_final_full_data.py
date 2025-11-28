#!/usr/bin/env python3
"""
Step 4.1: Train Final Models on Full Data

Purpose: Train final transfer learning models on full datasets for survival analysis.

Protocol:
- TCGA model: Pretrain on full ORIEN → Fine-tune on full TCGA
- ORIEN model: Pretrain on full TCGA → Fine-tune on full ORIEN
- Extract risk scores for ALL patients
- Calculate bootstrap-corrected C-index
- Save models and risk scores for Step 4.2 survival analysis

Configuration: k=155 (87 consensus genes)
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils import resample
from lifelines.utils import concordance_index

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.elastic_deepsurv import ElasticDeepSurv
from src.data.dataset import SurvivalDataset
from src.utils.batch_samplers import StratifiedBatchSampler

# ============================================================
# Configuration for k=155
# ============================================================

K_VALUE = 155
CONSENSUS_GENES_FILE = f'results_v2/02_biomarker_discovery/k_selection_with_tuning/k{K_VALUE}/consensus_genes/consensus_genes.txt'
OUTPUT_DIR = Path(f'results_v2/04_final_models/k{K_VALUE}')
N_BOOTSTRAP = 1000

# Fine-tuning LR multiplier (from Step 3 optimization)
LR_MULTIPLIER = 0.75

# Best hyperparameters from k=155 tuning
TCGA_CONFIG = {
    'hidden_sizes': [32],
    'dropout': 0.394,
    'batch_norm': False,
    'learning_rate': 8.87e-05,
    'batch_size': 64,
    'epochs': 250,
    'alpha': 0.000730,
    'l1_ratio': 0.857,
    'activation': 'relu',
    'weight_init': 'kaiming_normal'
}

ORIEN_CONFIG = {
    'hidden_sizes': [48],
    'dropout': 0.433,
    'batch_norm': True,
    'learning_rate': 7.59e-05,
    'batch_size': 32,
    'epochs': 250,
    'alpha': 0.000388,
    'l1_ratio': 0.520,
    'activation': 'elu',
    'weight_init': 'kaiming_normal'
}

# ============================================================
# Data Loading
# ============================================================

def load_consensus_genes(filepath):
    """Load consensus genes from file"""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_data(consensus_genes):
    """Load and filter data to consensus genes"""
    logger.info("Loading data files...")
    
    # Load expression data
    tcga_expr = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
    
    # Load survival data
    tcga_surv = pd.read_csv("data/raw/surv_tcga.csv")
    orien_surv = pd.read_csv("data/raw/surv_orien_update.csv")
    
    # Filter to consensus genes
    logger.info(f"Filtering to {len(consensus_genes)} consensus genes...")
    
    available_tcga = [g for g in consensus_genes if g in tcga_expr.index]
    available_orien = [g for g in consensus_genes if g in orien_expr.index]
    common_genes = sorted(list(set(available_tcga) & set(available_orien)))
    
    logger.info(f"  Using {len(common_genes)} genes available in both cohorts")
    
    tcga_expr = tcga_expr.loc[common_genes]
    orien_expr = orien_expr.loc[common_genes]
    
    # Match samples between expression and survival data
    def match_samples(expr_df, surv_df, cohort_name):
        expr_samples = set(expr_df.columns)
        surv_samples = set(surv_df['sampleID'])
        matched = sorted(list(expr_samples.intersection(surv_samples)))
        
        logger.info(f"  {cohort_name}: {len(matched)} samples matched")
        
        expr_df = expr_df[matched]
        surv_df = surv_df[surv_df['sampleID'].isin(matched)].set_index('sampleID')
        surv_df = surv_df.loc[matched]  # Ensure same order
        
        return expr_df, surv_df
    
    tcga_expr, tcga_surv = match_samples(tcga_expr, tcga_surv, 'TCGA')
    orien_expr, orien_surv = match_samples(orien_expr, orien_surv, 'ORIEN')
    
    # Standardize (Z-score per gene)
    logger.info("Standardizing expression data...")
    
    def standardize(expr_df):
        mean = expr_df.mean(axis=1)
        std = expr_df.std(axis=1).replace(0, 1)
        return expr_df.subtract(mean, axis=0).divide(std, axis=0)
    
    tcga_expr = standardize(tcga_expr)
    orien_expr = standardize(orien_expr)
    
    # Prepare data dictionaries
    tcga_data = {
        'X': tcga_expr,  # DataFrame (genes × samples)
        'y_time': tcga_surv['time'].values.astype(np.float32),
        'y_event': tcga_surv['event'].values.astype(np.int32),
        'sample_ids': tcga_expr.columns.tolist(),
        'surv_df': tcga_surv[['time', 'event']]
    }
    
    orien_data = {
        'X': orien_expr,
        'y_time': orien_surv['time'].values.astype(np.float32),
        'y_event': orien_surv['event'].values.astype(np.int32),
        'sample_ids': orien_expr.columns.tolist(),
        'surv_df': orien_surv[['time', 'event']]
    }
    
    logger.info(f"  TCGA: {tcga_data['X'].shape[1]} samples × {tcga_data['X'].shape[0]} genes")
    logger.info(f"  ORIEN: {orien_data['X'].shape[1]} samples × {orien_data['X'].shape[0]} genes")
    
    return tcga_data, orien_data


# ============================================================
# Model Training Functions
# ============================================================

def train_neural_network(X_train, y_time, y_event, config, device='cuda', pretrained_model=None):
    """
    Train neural network survival model.
    
    Args:
        X_train: DataFrame (genes × samples)
        y_time: survival times
        y_event: event indicators
        config: hyperparameters dict
        device: torch device
        pretrained_model: optional pretrained model for fine-tuning
    
    Returns:
        trained model
    """
    # Create survival DataFrame
    surv_df = pd.DataFrame({
        'time': y_time,
        'event': y_event
    }, index=X_train.columns)
    
    n_samples = X_train.shape[1]
    n_features = X_train.shape[0]
    
    logger.info(f"  Dataset: {n_samples} samples × {n_features} features")
    logger.info(f"  Events: {surv_df['event'].sum()}/{len(surv_df)} ({100*surv_df['event'].mean():.1f}%)")
    
    # Create dataset
    dataset = SurvivalDataset(X_train, surv_df)
    
    # Create model or use pretrained
    if pretrained_model is not None:
        model = pretrained_model
        logger.info(f"  Using pretrained model, fine-tuning with LR={config['learning_rate']:.6f}")
    else:
        model = ElasticDeepSurv(
            n_features=n_features,
            hidden_sizes=config['hidden_sizes'],
            dropout=config['dropout'],
            batch_norm=config['batch_norm'],
            alpha=config['alpha'],
            l1_ratio=config['l1_ratio'],
            activation=config.get('activation', 'relu'),
            weight_init=config.get('weight_init', 'kaiming_normal')
        ).to(device)
        logger.info(f"  Created new model with architecture {config['hidden_sizes']}")
    
    model = model.to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # DataLoader with stratified sampling for larger cohorts
    if n_samples >= 500:
        sampler = StratifiedBatchSampler(
            dataset.y_event,
            batch_size=config['batch_size'],
            min_events_per_batch=2,
            shuffle=True,
            drop_last=False
        )
        loader = DataLoader(dataset, batch_sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in loader:
            batch_x = batch['features'].to(device)
            batch_time = batch['time'].to(device)
            batch_event = batch['event'].to(device)
            
            optimizer.zero_grad()
            risk_scores = model(batch_x)
            loss = model.compute_loss(risk_scores, batch_time, batch_event)
            
            if loss is not None and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
        
        # Log every 50 epochs
        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"    Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")
    
    return model


def extract_risk_scores(model, X, device='cuda'):
    """
    Extract risk scores from trained model.
    
    Args:
        model: trained ElasticDeepSurv model
        X: DataFrame (genes × samples)
        device: torch device
    
    Returns:
        numpy array of risk scores
    """
    model.eval()
    
    # Transpose to samples × genes
    X_array = X.T.values.astype(np.float32)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_array).to(device)
        risk_scores = model(X_tensor).cpu().numpy().flatten()
    
    return risk_scores


# ============================================================
# Bootstrap C-index Calculation
# ============================================================

def bootstrap_cindex(y_event, y_time, risk_scores, n_bootstrap=1000):
    """
    Calculate bootstrap-corrected C-index.
    
    Reference: Harrell et al. (1996) - optimism-corrected bootstrap
    """
    # Apparent C-index
    apparent_cindex = concordance_index(y_time, -risk_scores, y_event)
    
    n_samples = len(risk_scores)
    optimism_scores = []
    
    logger.info(f"  Running {n_bootstrap} bootstrap iterations...")
    
    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            logger.info(f"    Bootstrap iteration {i+1}/{n_bootstrap}")
        
        # Resample with replacement
        boot_indices = resample(
            np.arange(n_samples),
            replace=True,
            random_state=i,
            n_samples=n_samples
        )
        
        # In-sample C-index
        c_boot_in = concordance_index(
            y_time[boot_indices],
            -risk_scores[boot_indices],
            y_event[boot_indices]
        )
        
        # Out-of-bag indices
        oob_indices = np.array([j for j in range(n_samples) if j not in set(boot_indices)])
        
        if len(oob_indices) > 10:
            c_boot_out = concordance_index(
                y_time[oob_indices],
                -risk_scores[oob_indices],
                y_event[oob_indices]
            )
            optimism_scores.append(c_boot_in - c_boot_out)
    
    # Average optimism and correction
    avg_optimism = np.mean(optimism_scores)
    optimism_ci = np.percentile(optimism_scores, [2.5, 97.5])
    corrected_cindex = apparent_cindex - avg_optimism
    corrected_ci = [apparent_cindex - optimism_ci[1], apparent_cindex - optimism_ci[0]]
    
    return {
        'apparent': float(apparent_cindex),
        'optimism': float(avg_optimism),
        'corrected': float(corrected_cindex),
        'corrected_ci_95': [float(corrected_ci[0]), float(corrected_ci[1])],
        'optimism_ci_95': [float(optimism_ci[0]), float(optimism_ci[1])]
    }


# ============================================================
# Main Training Function
# ============================================================

def train_all_models():
    """
    Train transfer learning models on full data.
    """
    logger.info("=" * 60)
    logger.info(f"Step 4.1: Training Final Models (k={K_VALUE})")
    logger.info("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load consensus genes
    logger.info(f"\nLoading consensus genes from k={K_VALUE}...")
    consensus_genes = load_consensus_genes(CONSENSUS_GENES_FILE)
    logger.info(f"  Loaded {len(consensus_genes)} genes")
    
    # Load data
    logger.info("\nLoading data...")
    tcga_data, orien_data = load_data(consensus_genes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  Device: {device}")
    
    all_results = {}
    
    # ============================================================
    # 1. ORIEN→TCGA Transfer Learning
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training ORIEN→TCGA Transfer Learning")
    logger.info("=" * 60)
    
    # Step 1a: Pretrain on full ORIEN using TCGA architecture
    logger.info("\n[1a] Pretraining on full ORIEN...")
    pretrain_config = ORIEN_CONFIG.copy()
    pretrain_config['hidden_sizes'] = TCGA_CONFIG['hidden_sizes']  # Use target architecture
    
    pretrain_model_orien = train_neural_network(
        orien_data['X'], orien_data['y_time'], orien_data['y_event'],
        pretrain_config, device
    )
    
    # Step 1b: Fine-tune on full TCGA
    logger.info("\n[1b] Fine-tuning on full TCGA...")
    finetune_config = TCGA_CONFIG.copy()
    finetune_config['learning_rate'] = ORIEN_CONFIG['learning_rate'] * LR_MULTIPLIER
    
    model_orien_to_tcga = train_neural_network(
        tcga_data['X'], tcga_data['y_time'], tcga_data['y_event'],
        finetune_config, device, pretrained_model=pretrain_model_orien
    )
    
    # Extract risk scores
    risk_scores_tcga = extract_risk_scores(model_orien_to_tcga, tcga_data['X'], device)
    
    # Bootstrap C-index
    logger.info("\n  Calculating bootstrap-corrected C-index...")
    bootstrap_results_tcga = bootstrap_cindex(
        tcga_data['y_event'], tcga_data['y_time'], risk_scores_tcga, N_BOOTSTRAP
    )
    
    logger.info(f"  Apparent C-index: {bootstrap_results_tcga['apparent']:.4f}")
    logger.info(f"  Optimism: {bootstrap_results_tcga['optimism']:.4f}")
    logger.info(f"  Corrected C-index: {bootstrap_results_tcga['corrected']:.4f} "
               f"({bootstrap_results_tcga['corrected_ci_95'][0]:.4f}-{bootstrap_results_tcga['corrected_ci_95'][1]:.4f})")
    
    # Save results
    all_results['TCGA_ORIENtoTCGA'] = {
        'method': 'ORIEN→TCGA Transfer',
        'cohort': 'TCGA',
        'bootstrap_results': bootstrap_results_tcga,
        'n_samples': tcga_data['X'].shape[1],
        'n_events': int(tcga_data['y_event'].sum())
    }
    
    # Save risk scores
    risk_df = pd.DataFrame({
        'sample_id': tcga_data['sample_ids'],
        'risk_score': risk_scores_tcga,
        'time': tcga_data['y_time'],
        'event': tcga_data['y_event']
    })
    risk_df.to_csv(OUTPUT_DIR / 'TCGA_ORIENtoTCGA_risk_scores.csv', index=False)
    
    # Save model
    torch.save({
        'model_state_dict': model_orien_to_tcga.state_dict(),
        'config': finetune_config,
        'bootstrap_results': bootstrap_results_tcga
    }, OUTPUT_DIR / 'TCGA_ORIENtoTCGA_model.pth')
    
    logger.info("  Saved model and risk scores")
    
    # ============================================================
    # 2. TCGA→ORIEN Transfer Learning
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training TCGA→ORIEN Transfer Learning")
    logger.info("=" * 60)
    
    # Step 2a: Pretrain on full TCGA using ORIEN architecture
    logger.info("\n[2a] Pretraining on full TCGA...")
    pretrain_config = TCGA_CONFIG.copy()
    pretrain_config['hidden_sizes'] = ORIEN_CONFIG['hidden_sizes']  # Use target architecture
    
    pretrain_model_tcga = train_neural_network(
        tcga_data['X'], tcga_data['y_time'], tcga_data['y_event'],
        pretrain_config, device
    )
    
    # Step 2b: Fine-tune on full ORIEN
    logger.info("\n[2b] Fine-tuning on full ORIEN...")
    finetune_config = ORIEN_CONFIG.copy()
    finetune_config['learning_rate'] = TCGA_CONFIG['learning_rate'] * LR_MULTIPLIER
    
    model_tcga_to_orien = train_neural_network(
        orien_data['X'], orien_data['y_time'], orien_data['y_event'],
        finetune_config, device, pretrained_model=pretrain_model_tcga
    )
    
    # Extract risk scores
    risk_scores_orien = extract_risk_scores(model_tcga_to_orien, orien_data['X'], device)
    
    # Bootstrap C-index
    logger.info("\n  Calculating bootstrap-corrected C-index...")
    bootstrap_results_orien = bootstrap_cindex(
        orien_data['y_event'], orien_data['y_time'], risk_scores_orien, N_BOOTSTRAP
    )
    
    logger.info(f"  Apparent C-index: {bootstrap_results_orien['apparent']:.4f}")
    logger.info(f"  Optimism: {bootstrap_results_orien['optimism']:.4f}")
    logger.info(f"  Corrected C-index: {bootstrap_results_orien['corrected']:.4f} "
               f"({bootstrap_results_orien['corrected_ci_95'][0]:.4f}-{bootstrap_results_orien['corrected_ci_95'][1]:.4f})")
    
    # Save results
    all_results['ORIEN_TCGAtoORIEN'] = {
        'method': 'TCGA→ORIEN Transfer',
        'cohort': 'ORIEN',
        'bootstrap_results': bootstrap_results_orien,
        'n_samples': orien_data['X'].shape[1],
        'n_events': int(orien_data['y_event'].sum())
    }
    
    # Save risk scores
    risk_df = pd.DataFrame({
        'sample_id': orien_data['sample_ids'],
        'risk_score': risk_scores_orien,
        'time': orien_data['y_time'],
        'event': orien_data['y_event']
    })
    risk_df.to_csv(OUTPUT_DIR / 'ORIEN_TCGAtoORIEN_risk_scores.csv', index=False)
    
    # Save model
    torch.save({
        'model_state_dict': model_tcga_to_orien.state_dict(),
        'config': finetune_config,
        'bootstrap_results': bootstrap_results_orien
    }, OUTPUT_DIR / 'ORIEN_TCGAtoORIEN_model.pth')
    
    logger.info("  Saved model and risk scores")
    
    # ============================================================
    # Save Summary
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    
    # Save all results
    with open(OUTPUT_DIR / 'bootstrap_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary table
    summary_data = []
    for key, result in all_results.items():
        summary_data.append({
            'Model': result['method'],
            'Cohort': result['cohort'],
            'N': result['n_samples'],
            'Events': result['n_events'],
            'Apparent_Cindex': result['bootstrap_results']['apparent'],
            'Optimism': result['bootstrap_results']['optimism'],
            'Corrected_Cindex': result['bootstrap_results']['corrected'],
            'CI_Lower': result['bootstrap_results']['corrected_ci_95'][0],
            'CI_Upper': result['bootstrap_results']['corrected_ci_95'][1]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / 'performance_summary.csv', index=False)
    
    logger.info(f"\nPerformance Summary:")
    logger.info("\n" + summary_df.to_string(index=False))
    
    logger.info("\n" + "=" * 60)
    logger.info("Step 4.1 Complete!")
    logger.info("=" * 60)
    logger.info(f"\nAll results saved to: {OUTPUT_DIR}")
    logger.info("\nGenerated files:")
    logger.info("  - bootstrap_results.json")
    logger.info("  - performance_summary.csv")
    logger.info("  - TCGA_ORIENtoTCGA_risk_scores.csv")
    logger.info("  - ORIEN_TCGAtoORIEN_risk_scores.csv")
    logger.info("  - TCGA_ORIENtoTCGA_model.pth")
    logger.info("  - ORIEN_TCGAtoORIEN_model.pth")
    
    return all_results


if __name__ == '__main__':
    train_all_models()
