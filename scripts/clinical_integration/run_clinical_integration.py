"""
Clinical Integration Cross-Cohort Validation

Phase 1: Gene + Clinical Integration for DeepSurv

This script:
1. Loads 58 consensus genes + 6 clinical variables (15 features after one-hot encoding)
2. Runs cross-cohort validation (ORIEN→TCGA and TCGA→ORIEN)
3. Computes C-index with multi-seed validation (5 seeds)
4. Uses existing k=90 hyperparameters as starting point
5. Generates comparison with gene-only baseline

Clinical variables (one-hot encoded):
- age: z-scored (1 feature)
- gender: binary (1 feature)  
- smoking: Never/Ever/Unknown (3 features)
- alcohol: Never/Ever/Unknown (3 features)
- N_stage: N0/N1/N2-3/Unknown (4 features)
- T_stage: T1-T2/T3-T4/Unknown (3 features)

Total: 58 genes + 15 clinical = 73 features

Methodology:
- Train on 100% source cohort (matches Cox comparison)
- Fixed epochs (from hyperparameter tuning CV)
- Standardization: fit on source, transform target

References:
- Katzman et al. (2018) DeepSurv - clinical variable integration
- Harrell (2015) - EPV guidelines for survival models
- Bernau et al. (2014) - Cross-study validation

Usage:
    python run_clinical_integration.py \
        --data_dir data \
        --consensus_genes consensus_genes.txt \
        --output_dir results_v2/03_clinical_integration \
        --epochs 200

Author: Phuong
Created: 2025
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import yaml
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

# Import from existing modules
from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv
from src.utils.batch_samplers import StratifiedBatchSampler

# Import clinical preprocessor
from clinical_preprocessor import ClinicalPreprocessor, IntegratedPreprocessor, load_clinical_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seeds for multi-seed validation (same as gene-only for fair comparison)
SEEDS = [42, 123, 456, 789, 1011]


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_dir: Path, consensus_gene_file: Path) -> Dict:
    """
    Load all required data: expression, survival, clinical, and consensus genes.
    
    Returns:
        Dict with all data components
    """
    logger.info("="*60)
    logger.info("Loading Data")
    logger.info("="*60)
    
    # Load expression data
    logger.info("\nLoading expression data...")
    tcga_expr = pd.read_csv(data_dir / "raw" / "tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv(data_dir / "raw" / "orien_batch_corrected.csv", index_col=0)
    logger.info(f"  TCGA: {tcga_expr.shape[0]} genes × {tcga_expr.shape[1]} samples")
    logger.info(f"  ORIEN: {orien_expr.shape[0]} genes × {orien_expr.shape[1]} samples")
    
    # Load survival data
    logger.info("\nLoading survival data...")
    tcga_surv = pd.read_csv(data_dir / "processed" / "surv_tcga_harmonized.csv", index_col=0)
    orien_surv = pd.read_csv(data_dir / "processed" / "surv_orien_harmonized.csv", index_col=0)
    logger.info(f"  TCGA: {len(tcga_surv)} samples, {tcga_surv['event'].sum()} events ({tcga_surv['event'].mean():.1%})")
    logger.info(f"  ORIEN: {len(orien_surv)} samples, {orien_surv['event'].sum()} events ({orien_surv['event'].mean():.1%})")
    
    # Load clinical data
    logger.info("\nLoading clinical data...")
    tcga_clinical = pd.read_csv(data_dir / "raw" / "clinical_tcga.csv")
    orien_clinical = pd.read_csv(data_dir / "raw" / "clinical_orien_updated.csv")
    logger.info(f"  TCGA: {tcga_clinical.shape}")
    logger.info(f"  ORIEN: {orien_clinical.shape}")
    
    # Load consensus genes
    logger.info(f"\nLoading consensus genes from {consensus_gene_file}...")
    with open(consensus_gene_file, 'r') as f:
        consensus_genes = [line.strip() for line in f if line.strip()]
    logger.info(f"  Loaded {len(consensus_genes)} consensus genes")
    
    # Filter expression to consensus genes
    available_tcga = [g for g in consensus_genes if g in tcga_expr.index]
    available_orien = [g for g in consensus_genes if g in orien_expr.index]
    
    if len(available_tcga) < len(consensus_genes):
        logger.warning(f"  TCGA: only {len(available_tcga)}/{len(consensus_genes)} genes available")
    if len(available_orien) < len(consensus_genes):
        logger.warning(f"  ORIEN: only {len(available_orien)}/{len(consensus_genes)} genes available")
    
    # Use intersection of available genes
    common_genes = sorted(list(set(available_tcga) & set(available_orien)))
    logger.info(f"  Using {len(common_genes)} common consensus genes")
    
    tcga_expr = tcga_expr.loc[common_genes]
    orien_expr = orien_expr.loc[common_genes]
    
    return {
        'tcga_expr': tcga_expr,
        'orien_expr': orien_expr,
        'tcga_surv': tcga_surv,
        'orien_surv': orien_surv,
        'tcga_clinical': tcga_clinical,
        'orien_clinical': orien_clinical,
        'consensus_genes': common_genes
    }


def load_hyperparameters(params_file: Path) -> Dict:
    """Load hyperparameters from JSON file."""
    with open(params_file, 'r') as f:
        params = json.load(f)
    return params


def parse_architecture(best_params: Dict) -> List[int]:
    """Parse hidden layer sizes from hyperparameter dict."""
    if 'architecture_2layer' in best_params:
        return [int(x) for x in best_params['architecture_2layer'].split('-')]
    elif 'architecture_3layer' in best_params:
        return [int(x) for x in best_params['architecture_3layer'].split('-')]
    elif 'layer1_size' in best_params:
        return [best_params['layer1_size']]
    else:
        raise ValueError(f"Cannot parse architecture from {best_params}")


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in train_loader:
        features = batch['features'].to(device)
        times = batch['time'].to(device)
        events = batch['event'].to(device)
        
        optimizer.zero_grad()
        log_hazards = model(features)
        loss = model.compute_loss(log_hazards, times, events)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate_model(model, data_loader, device) -> float:
    """Evaluate model and return C-index."""
    model.eval()
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            times = batch['time'].cpu().numpy()
            events = batch['event'].cpu().numpy()
            
            log_hazards = model(features)
            risks = torch.exp(log_hazards).squeeze().cpu().numpy()
            
            if np.isscalar(risks):
                risks = np.array([risks])
            
            all_risks.extend(risks)
            all_times.extend(times)
            all_events.extend(events)
    
    all_risks = np.array(all_risks)
    all_times = np.array(all_times)
    all_events = np.array(all_events)
    
    c_index = concordance_index(all_times, -all_risks, all_events)
    
    return c_index


class ClinicalSurvivalDataset(torch.utils.data.Dataset):
    """
    Dataset for combined gene + clinical features.
    
    Unlike SurvivalDataset which expects genes × samples format,
    this takes samples × features format directly.
    """
    
    def __init__(self, features_df: pd.DataFrame, survival_df: pd.DataFrame):
        """
        Args:
            features_df: Combined features (samples × features)
            survival_df: Survival data with 'time' and 'event' columns
        """
        # Align samples
        common_samples = list(set(features_df.index) & set(survival_df.index))
        common_samples = sorted(common_samples)
        
        self.features = features_df.loc[common_samples]
        self.survival = survival_df.loc[common_samples]
        
        self.X = self.features.values.astype(np.float32)
        self.y_time = self.survival['time'].values.astype(np.float32)
        self.y_event = self.survival['event'].values.astype(np.float32)
        
        self.sample_ids = common_samples
        self.feature_names = features_df.columns.tolist()
        self.n_features = self.X.shape[1]
        
        logger.info(f"ClinicalSurvivalDataset: {len(self)} samples, {self.n_features} features")
        logger.info(f"  Event rate: {self.y_event.mean():.1%}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': torch.from_numpy(self.X[idx]),
            'time': torch.tensor(self.y_time[idx]),
            'event': torch.tensor(self.y_event[idx]),
            'sample_id': self.sample_ids[idx]
        }


def train_and_evaluate_single_seed(
    source_cohort: str,
    target_cohort: str,
    data_dict: Dict,
    hyperparams: Dict,
    seed: int,
    epochs: int,
    device: str,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Train on source cohort with clinical integration, evaluate on target.
    
    Args:
        source_cohort: 'tcga' or 'orien'
        target_cohort: 'tcga' or 'orien'
        data_dict: Dict containing all data
        hyperparams: Model hyperparameters
        seed: Random seed
        epochs: Number of training epochs
        device: Device for training
        output_dir: Optional directory for saving training curves
        
    Returns:
        Dict with training results
    """
    set_seed(seed)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Seed {seed}: {source_cohort.upper()} → {target_cohort.upper()}")
    logger.info(f"{'='*60}")
    
    # Get data for source and target
    if source_cohort.lower() == 'tcga':
        source_expr = data_dict['tcga_expr']
        source_surv = data_dict['tcga_surv']
        source_clinical = data_dict['tcga_clinical']
        target_expr = data_dict['orien_expr']
        target_surv = data_dict['orien_surv']
        target_clinical = data_dict['orien_clinical']
    else:
        source_expr = data_dict['orien_expr']
        source_surv = data_dict['orien_surv']
        source_clinical = data_dict['orien_clinical']
        target_expr = data_dict['tcga_expr']
        target_surv = data_dict['tcga_surv']
        target_clinical = data_dict['tcga_clinical']
    
    # Create config for preprocessor
    config = {
        'data': {
            'min_variance_percentile': 0,
            'standardize': True
        }
    }
    
    # Initialize integrated preprocessor
    preprocessor = IntegratedPreprocessor(config)
    
    # Fit on source, transform both
    source_features = preprocessor.fit_transform(source_expr, source_clinical, source_cohort)
    target_features = preprocessor.transform(target_expr, target_clinical, target_cohort)
    
    feature_info = preprocessor.get_feature_names()
    n_genes = len(feature_info['genes'])
    n_clinical = len(feature_info['clinical'])
    n_total = len(feature_info['all'])
    
    logger.info(f"\nFeature summary: {n_genes} genes + {n_clinical} clinical = {n_total} total")
    
    # Create datasets
    source_dataset = ClinicalSurvivalDataset(source_features, source_surv)
    target_dataset = ClinicalSurvivalDataset(target_features, target_surv)
    
    # Parse hyperparameters
    best_params = hyperparams['best_params']
    hidden_sizes = parse_architecture(best_params)
    
    # Scale first layer size proportionally to new feature count
    # Original: 58 genes, New: 73 features (58 genes + 15 clinical)
    # This is a heuristic - may need tuning
    original_n_features = hyperparams.get('m', 58)
    scale_factor = n_total / original_n_features
    
    # Optionally scale first hidden layer (conservative approach: don't scale)
    # hidden_sizes[0] = int(hidden_sizes[0] * scale_factor)
    
    logger.info(f"\nModel architecture:")
    logger.info(f"  Input features: {n_total}")
    logger.info(f"  Hidden layers: {hidden_sizes}")
    logger.info(f"  Hyperparameters from k={hyperparams.get('k', 'unknown')}")
    
    # Create data loaders
    batch_size = best_params['batch_size']
    
    if len(source_dataset) < 400:
        source_loader = DataLoader(
            source_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
    else:
        source_sampler = StratifiedBatchSampler(
            events=source_dataset.y_event,
            batch_size=batch_size,
            min_events_per_batch=2,
            shuffle=True
        )
        source_loader = DataLoader(
            source_dataset,
            batch_sampler=source_sampler,
            num_workers=0
        )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = ElasticDeepSurv(
        n_features=n_total,
        hidden_sizes=hidden_sizes,
        dropout=best_params['dropout'],
        activation=best_params['activation'],
        batch_norm=best_params['batch_norm'],
        weight_init=best_params.get('weight_init', 'kaiming_normal'),
        l1_ratio=best_params['l1_ratio'],
        alpha=best_params['alpha']
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    
    # Training loop
    logger.info(f"\nTraining for {epochs} epochs...")
    training_history = []
    best_test_cindex = 0.0
    best_test_epoch = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, source_loader, optimizer, device)
        
        # Evaluate every 10 epochs or at key points
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            train_cindex = evaluate_model(model, source_loader, device)
            test_cindex = evaluate_model(model, target_loader, device)
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_cindex': train_cindex,
                'test_cindex': test_cindex
            })
            
            if test_cindex > best_test_cindex:
                best_test_cindex = test_cindex
                best_test_epoch = epoch + 1
            
            if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                logger.info(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, "
                           f"Train={train_cindex:.4f}, Test={test_cindex:.4f}")
    
    # Final evaluation
    final_train_cindex = evaluate_model(model, source_loader, device)
    final_test_cindex = evaluate_model(model, target_loader, device)
    
    logger.info(f"\nFinal Results:")
    logger.info(f"  Train C-index: {final_train_cindex:.4f}")
    logger.info(f"  Test C-index: {final_test_cindex:.4f}")
    logger.info(f"  Best Test C-index: {best_test_cindex:.4f} at epoch {best_test_epoch}")
    
    # Save training curve if output_dir provided
    if output_dir is not None and training_history:
        save_training_curve(
            training_history,
            source_cohort,
            target_cohort,
            seed,
            output_dir
        )
    
    return {
        'seed': seed,
        'source': source_cohort,
        'target': target_cohort,
        'train_cindex': final_train_cindex,
        'test_cindex': final_test_cindex,
        'best_test_cindex': best_test_cindex,
        'best_test_epoch': best_test_epoch,
        'n_features': n_total,
        'n_genes': n_genes,
        'n_clinical': n_clinical,
        'epochs': epochs,
        'training_history': training_history
    }


def save_training_curve(
    history: List[Dict],
    source: str,
    target: str,
    seed: int,
    output_dir: Path
):
    """Save training curve plot."""
    plot_dir = output_dir / "training_curves"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = [h['epoch'] for h in history]
    train_cindices = [h['train_cindex'] for h in history]
    test_cindices = [h['test_cindex'] for h in history]
    losses = [h['train_loss'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(epochs, losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.set_title(f'{source.upper()}→{target.upper()} Loss (Seed {seed})')
    ax1.grid(True, alpha=0.3)
    
    # C-index
    ax2.plot(epochs, train_cindices, 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, test_cindices, 'r-', label='Test', linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('C-index')
    ax2.set_title(f'{source.upper()}→{target.upper()} C-index (Seed {seed})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / f'{source}_to_{target}_seed{seed}.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_cross_cohort_validation(
    data_dict: Dict,
    hyperparams_tcga: Dict,
    hyperparams_orien: Dict,
    epochs: int,
    seeds: List[int],
    output_dir: Path,
    device: str
) -> Dict:
    """
    Run full cross-cohort validation in both directions.
    
    Returns:
        Dict with aggregated results
    """
    logger.info("\n" + "="*70)
    logger.info("CROSS-COHORT VALIDATION WITH CLINICAL INTEGRATION")
    logger.info("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Direction 1: ORIEN → TCGA
    logger.info("\n" + "="*70)
    logger.info("Direction 1: ORIEN → TCGA")
    logger.info("="*70)
    
    orien_to_tcga_results = []
    for seed in seeds:
        result = train_and_evaluate_single_seed(
            source_cohort='orien',
            target_cohort='tcga',
            data_dict=data_dict,
            hyperparams=hyperparams_orien,
            seed=seed,
            epochs=epochs,
            device=device,
            output_dir=output_dir
        )
        orien_to_tcga_results.append(result)
    
    # Direction 2: TCGA → ORIEN
    logger.info("\n" + "="*70)
    logger.info("Direction 2: TCGA → ORIEN")
    logger.info("="*70)
    
    tcga_to_orien_results = []
    for seed in seeds:
        result = train_and_evaluate_single_seed(
            source_cohort='tcga',
            target_cohort='orien',
            data_dict=data_dict,
            hyperparams=hyperparams_tcga,
            seed=seed,
            epochs=epochs,
            device=device,
            output_dir=output_dir
        )
        tcga_to_orien_results.append(result)
    
    # Aggregate results
    o2t_test = [r['test_cindex'] for r in orien_to_tcga_results]
    t2o_test = [r['test_cindex'] for r in tcga_to_orien_results]
    
    o2t_mean = np.mean(o2t_test)
    o2t_std = np.std(o2t_test)
    t2o_mean = np.mean(t2o_test)
    t2o_std = np.std(t2o_test)
    
    mean_bidirectional = (o2t_mean + t2o_mean) / 2
    mean_bidirectional_std = np.sqrt((o2t_std**2 + t2o_std**2) / 4)
    
    # Get feature counts from first result
    n_features = orien_to_tcga_results[0]['n_features']
    n_genes = orien_to_tcga_results[0]['n_genes']
    n_clinical = orien_to_tcga_results[0]['n_clinical']
    
    summary = {
        'n_features': n_features,
        'n_genes': n_genes,
        'n_clinical': n_clinical,
        'n_seeds': len(seeds),
        'epochs': epochs,
        'orien_to_tcga': {
            'mean': o2t_mean,
            'std': o2t_std,
            'all_cindices': o2t_test
        },
        'tcga_to_orien': {
            'mean': t2o_mean,
            'std': t2o_std,
            'all_cindices': t2o_test
        },
        'mean_bidirectional': mean_bidirectional,
        'mean_bidirectional_std': mean_bidirectional_std,
        'timestamp': datetime.now().isoformat(),
        'per_seed_results': {
            'orien_to_tcga': orien_to_tcga_results,
            'tcga_to_orien': tcga_to_orien_results
        }
    }
    
    return summary


def print_comparison_table(clinical_results: Dict, gene_only_baseline: Dict = None):
    """Print comparison table of results."""
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)
    
    logger.info(f"\nClinical Integration Results:")
    logger.info(f"  Features: {clinical_results['n_genes']} genes + {clinical_results['n_clinical']} clinical = {clinical_results['n_features']} total")
    logger.info(f"  Seeds: {clinical_results['n_seeds']}")
    logger.info(f"  Epochs: {clinical_results['epochs']}")
    
    logger.info(f"\n  ORIEN → TCGA: {clinical_results['orien_to_tcga']['mean']:.4f} ± {clinical_results['orien_to_tcga']['std']:.4f}")
    logger.info(f"  TCGA → ORIEN: {clinical_results['tcga_to_orien']['mean']:.4f} ± {clinical_results['tcga_to_orien']['std']:.4f}")
    logger.info(f"  Mean Bidirectional: {clinical_results['mean_bidirectional']:.4f} ± {clinical_results['mean_bidirectional_std']:.4f}")
    
    if gene_only_baseline:
        logger.info(f"\nGene-Only Baseline (for comparison):")
        logger.info(f"  ORIEN → TCGA: {gene_only_baseline.get('orien_to_tcga', 'N/A')}")
        logger.info(f"  TCGA → ORIEN: {gene_only_baseline.get('tcga_to_orien', 'N/A')}")
        logger.info(f"  Mean Bidirectional: {gene_only_baseline.get('mean_bidirectional', 'N/A')}")
        
        if 'mean_bidirectional' in gene_only_baseline:
            improvement = clinical_results['mean_bidirectional'] - gene_only_baseline['mean_bidirectional']
            logger.info(f"\n  Improvement: {improvement:+.4f} ({improvement/gene_only_baseline['mean_bidirectional']*100:+.1f}%)")
    
    logger.info("\n" + "="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Integration Cross-Cohort Validation')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--consensus_genes', type=str, default='consensus_genes.txt',
                        help='Path to consensus genes file')
    parser.add_argument('--tcga_params', type=str, default='best_params_tcga.json',
                        help='Path to TCGA hyperparameters')
    parser.add_argument('--orien_params', type=str, default='best_params_orien.json',
                        help='Path to ORIEN hyperparameters')
    parser.add_argument('--output_dir', type=str, default='results_v2/03_clinical_integration',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='Random seeds')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("="*70)
    logger.info("CLINICAL INTEGRATION CROSS-COHORT VALIDATION")
    logger.info("="*70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Consensus genes: {args.consensus_genes}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Device: {device}")
    logger.info("="*70)
    
    # Load data
    data_dict = load_data(data_dir, Path(args.consensus_genes))
    
    # Load hyperparameters
    logger.info("\nLoading hyperparameters...")
    hyperparams_tcga = load_hyperparameters(Path(args.tcga_params))
    hyperparams_orien = load_hyperparameters(Path(args.orien_params))
    logger.info(f"  TCGA: k={hyperparams_tcga.get('k')}, CV C-index={hyperparams_tcga.get('best_cv_cindex', 'N/A'):.4f}")
    logger.info(f"  ORIEN: k={hyperparams_orien.get('k')}, CV C-index={hyperparams_orien.get('best_cv_cindex', 'N/A'):.4f}")
    
    # Run cross-cohort validation
    results = run_cross_cohort_validation(
        data_dict=data_dict,
        hyperparams_tcga=hyperparams_tcga,
        hyperparams_orien=hyperparams_orien,
        epochs=args.epochs,
        seeds=args.seeds,
        output_dir=output_dir,
        device=device
    )
    
    # Gene-only baseline for comparison
    gene_only_baseline = {
        'orien_to_tcga': 0.6803,
        'tcga_to_orien': 0.6231,
        'mean_bidirectional': 0.6517
    }
    
    # Print results
    print_comparison_table(results, gene_only_baseline)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary (without training history for cleaner JSON)
    summary_for_save = {k: v for k, v in results.items() if k != 'per_seed_results'}
    summary_for_save['gene_only_baseline'] = gene_only_baseline
    
    with open(output_dir / 'clinical_integration_results.json', 'w') as f:
        json.dump(summary_for_save, f, indent=2)
    
    # Save detailed results
    with open(output_dir / 'clinical_integration_detailed.json', 'w') as f:
        # Convert training history to serializable format
        results_serializable = results.copy()
        for direction in ['orien_to_tcga', 'tcga_to_orien']:
            if direction in results_serializable.get('per_seed_results', {}):
                for r in results_serializable['per_seed_results'][direction]:
                    # Training history is already list of dicts, should be serializable
                    pass
        json.dump(results_serializable, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - clinical_integration_results.json (summary)")
    logger.info(f"  - clinical_integration_detailed.json (full results)")
    logger.info(f"  - training_curves/ (plots)")
    
    return results


if __name__ == '__main__':
    main()
