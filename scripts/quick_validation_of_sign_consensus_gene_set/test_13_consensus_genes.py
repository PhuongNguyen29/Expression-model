"""
Quick Diagnostic: Test Consensus Genes (Cox ∩ DeepSurv overlap)

Purpose: Quick test to determine if consensus gene signature has signal
         No hyperparameter tuning - uses conservative defaults

Usage:
    python test_13_consensus_genes.py --data_dir data --gene_file sign_consistent_genes_13.txt --epochs 150
"""

import sys
from pathlib import Path

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

from src.data.preprocessor import GeneExpressionPreprocessor
from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv
from lifelines.utils import concordance_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seeds for multi-seed validation
SEEDS = [42, 123, 456, 789, 1011]

# Conservative hyperparameters (no tuning needed)
CONSERVATIVE_PARAMS = {
    'hidden_sizes': [24],      # Small: 13 → 32 → 1
    'dropout': 0.4,            # Conservative dropout
    'alpha': 0.001,            # Moderate regularization
    'l1_ratio': 0.7,           # Sparse
    'learning_rate': 0.0001,   # Safe learning rate
    'batch_size': 32,          # Standard
    'activation': 'elu',
    'batch_norm': True,
    'weight_init': 'kaiming_normal'
}


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_genes_from_file(gene_file: Path) -> list:
    """Load gene list from text file (one Ensembl ID per line)."""
    with open(gene_file, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(genes)} genes from {gene_file}")
    return genes


def verify_genes_in_data(gene_list: list, expr_df: pd.DataFrame) -> list:
    """Verify which genes are present in expression data."""
    available = [g for g in gene_list if g in expr_df.index]
    missing = [g for g in gene_list if g not in expr_df.index]
    
    if missing:
        logger.warning(f"Genes not in expression data: {missing}")
    
    logger.info(f"Available genes: {len(available)}/{len(gene_list)}")
    return available


def load_data(data_dir: Path):
    """Load expression and survival data."""
    logger.info("Loading expression and survival data...")
    
    tcga_expr = pd.read_csv(data_dir / "raw" / "tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv(data_dir / "raw" / "orien_batch_corrected.csv", index_col=0)
    tcga_surv = pd.read_csv(data_dir / "processed" / "surv_tcga_harmonized.csv", index_col=0)
    orien_surv = pd.read_csv(data_dir / "processed" / "surv_orien_harmonized.csv", index_col=0)
    
    logger.info(f"  TCGA: {tcga_expr.shape[1]} samples, {tcga_expr.shape[0]} genes")
    logger.info(f"  ORIEN: {orien_expr.shape[1]} samples, {orien_expr.shape[0]} genes")
    
    return {
        'tcga_expr': tcga_expr,
        'orien_expr': orien_expr,
        'tcga_surv': tcga_surv,
        'orien_surv': orien_surv
    }


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


def evaluate_model(model, data_loader, device):
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


def run_cross_cohort_validation(
    source_cohort: str,
    target_cohort: str,
    consensus_genes: list,
    data_dict: dict,
    seed: int,
    epochs: int = 150,
    device: str = None
) -> dict:
    """
    Train on 100% SOURCE → Test on 100% TARGET
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    set_seed(seed)
    
    logger.info(f"\n--- Seed {seed}: {source_cohort.upper()} → {target_cohort.upper()} ---")
    
    # Get data
    if source_cohort.lower() == 'tcga':
        source_expr = data_dict['tcga_expr'].loc[consensus_genes]
        source_surv = data_dict['tcga_surv']
        target_expr = data_dict['orien_expr'].loc[consensus_genes]
        target_surv = data_dict['orien_surv']
    else:
        source_expr = data_dict['orien_expr'].loc[consensus_genes]
        source_surv = data_dict['orien_surv']
        target_expr = data_dict['tcga_expr'].loc[consensus_genes]
        target_surv = data_dict['tcga_surv']
    
    # Preprocess
    config = {
        'data': {
            'min_variance_percentile': 0,
            'standardize': True
        }
    }
    
    preprocessor = GeneExpressionPreprocessor(config)
    source_processed = preprocessor.fit_transform_single_cohort(source_expr, f'{source_cohort}_train')
    target_processed = preprocessor.transform_single_cohort(target_expr)
    
    # Create datasets
    source_dataset = SurvivalDataset(source_processed, source_surv)
    target_dataset = SurvivalDataset(target_processed, target_surv)
    
    source_loader = DataLoader(source_dataset, batch_size=CONSERVATIVE_PARAMS['batch_size'], shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=CONSERVATIVE_PARAMS['batch_size'], shuffle=False)
    
    # Create model
    n_features = len(consensus_genes)
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=CONSERVATIVE_PARAMS['hidden_sizes'],
        dropout=CONSERVATIVE_PARAMS['dropout'],
        activation=CONSERVATIVE_PARAMS['activation'],
        batch_norm=CONSERVATIVE_PARAMS['batch_norm'],
        weight_init=CONSERVATIVE_PARAMS['weight_init'],
        l1_ratio=CONSERVATIVE_PARAMS['l1_ratio'],
        alpha=CONSERVATIVE_PARAMS['alpha']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONSERVATIVE_PARAMS['learning_rate'])
    
    # Train
    for epoch in range(epochs):
        train_loss = train_epoch(model, source_loader, optimizer, device)
        
        if (epoch + 1) % 50 == 0:
            train_cindex = evaluate_model(model, source_loader, device)
            test_cindex = evaluate_model(model, target_loader, device)
            logger.info(f"  Epoch {epoch+1}: Train C-index={train_cindex:.4f}, Test C-index={test_cindex:.4f}")
    
    # Final evaluation
    final_train_cindex = evaluate_model(model, source_loader, device)
    final_test_cindex = evaluate_model(model, target_loader, device)
    
    return {
        'seed': seed,
        'source': source_cohort,
        'target': target_cohort,
        'train_cindex': final_train_cindex,
        'test_cindex': final_test_cindex,
        'n_genes': n_features
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick test of consensus genes')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--gene_file', type=str, required=True, 
                        help='Path to gene list file (one Ensembl ID per line)')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs')
    parser.add_argument('--output_dir', type=str, default='results_v2/09_consensus_13_genes', 
                        help='Output directory')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    gene_file = Path(args.gene_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("QUICK DIAGNOSTIC: CONSENSUS GENES (Cox ∩ DeepSurv)")
    logger.info("=" * 70)
    logger.info(f"Gene file: {gene_file}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Parameters: {CONSERVATIVE_PARAMS}")
    logger.info("=" * 70)
    
    # Load genes from file
    consensus_genes = load_genes_from_file(gene_file)
    
    # Load data
    data_dict = load_data(data_dir)
    
    # Verify genes exist in data
    available_genes = verify_genes_in_data(consensus_genes, data_dict['tcga_expr'])
    
    if len(available_genes) < 5:
        logger.error(f"Only {len(available_genes)} genes available. Need at least 5.")
        return
    
    logger.info(f"\nProceeding with {len(available_genes)} genes")
    logger.info(f"Genes: {available_genes}")
    
    # Run cross-cohort validation
    o2t_results = []
    t2o_results = []
    
    # ORIEN → TCGA
    logger.info("\n" + "=" * 70)
    logger.info("Direction: ORIEN → TCGA")
    logger.info("=" * 70)
    
    for seed in SEEDS:
        result = run_cross_cohort_validation(
            source_cohort='orien',
            target_cohort='tcga',
            consensus_genes=available_genes,
            data_dict=data_dict,
            seed=seed,
            epochs=args.epochs
        )
        o2t_results.append(result)
    
    # TCGA → ORIEN
    logger.info("\n" + "=" * 70)
    logger.info("Direction: TCGA → ORIEN")
    logger.info("=" * 70)
    
    for seed in SEEDS:
        result = run_cross_cohort_validation(
            source_cohort='tcga',
            target_cohort='orien',
            consensus_genes=available_genes,
            data_dict=data_dict,
            seed=seed,
            epochs=args.epochs
        )
        t2o_results.append(result)
    
    # Aggregate results
    o2t_test = [r['test_cindex'] for r in o2t_results]
    t2o_test = [r['test_cindex'] for r in t2o_results]
    
    o2t_mean = np.mean(o2t_test)
    o2t_std = np.std(o2t_test)
    t2o_mean = np.mean(t2o_test)
    t2o_std = np.std(t2o_test)
    mean_bidir = (o2t_mean + t2o_mean) / 2
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info(f"RESULTS SUMMARY: {len(available_genes)} CONSENSUS GENES")
    logger.info("=" * 70)
    logger.info(f"Genes used: {len(available_genes)}")
    logger.info(f"")
    logger.info(f"ORIEN → TCGA: {o2t_mean:.4f} ± {o2t_std:.4f}")
    logger.info(f"TCGA → ORIEN: {t2o_mean:.4f} ± {t2o_std:.4f}")
    logger.info(f"Mean Bidirectional: {mean_bidir:.4f}")
    logger.info("=" * 70)
    
    # Comparison table
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON WITH OTHER CONFIGURATIONS")
    logger.info("=" * 70)
    logger.info(f"{'Model':<25} {'Genes':<10} {'O→T':<12} {'T→O':<12} {'Mean':<10}")
    logger.info("-" * 70)
    logger.info(f"{'Cox-Lasso':<25} {'20':<10} {'0.72':<12} {'0.68':<12} {'0.70':<10}")
    logger.info(f"{'DeepSurv (IG genes)':<25} {'58':<10} {'0.68':<12} {'0.62':<12} {'0.65':<10}")
    logger.info(f"{'DeepSurv (Cox genes)':<25} {'20':<10} {'0.637':<12} {'0.562':<12} {'0.60':<10}")
    logger.info(f"{'DeepSurv (Consensus)':<25} {f'{len(available_genes)}':<10} {f'{o2t_mean:.3f}':<12} {f'{t2o_mean:.3f}':<12} {f'{mean_bidir:.3f}':<10}")
    logger.info("=" * 70)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_genes': len(available_genes),
        'genes_ensembl': available_genes,
        'gene_file': str(gene_file),
        'parameters': CONSERVATIVE_PARAMS,
        'epochs': args.epochs,
        'orien_to_tcga': {
            'mean': float(o2t_mean),
            'std': float(o2t_std),
            'per_seed': o2t_test
        },
        'tcga_to_orien': {
            'mean': float(t2o_mean),
            'std': float(t2o_std),
            'per_seed': t2o_test
        },
        'mean_bidirectional': float(mean_bidir)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir / 'results.json'}")
    
    # Interpretation
    logger.info("\n" + "=" * 70)
    logger.info("INTERPRETATION")
    logger.info("=" * 70)
    
    if mean_bidir >= 0.65:
        logger.info("✓ Consensus genes achieve SIMILAR performance to 58 genes")
        logger.info("  → Parsimonious signature is viable")
        logger.info("  → Consider: 'Multi-method consensus identifies compact OSCC signature'")
    elif mean_bidir >= 0.60:
        logger.info("~ Consensus genes achieve MODERATE performance")
        logger.info("  → Some signal present but weaker than 58 genes")
        logger.info("  → Consider: Methods identify complementary prognostic genes")
    else:
        logger.info("✗ Consensus genes achieve LOWER performance than expected")
        logger.info("  → Overlap genes alone insufficient")
        logger.info("  → Consider: Cox and DeepSurv capture different biology")
    
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
