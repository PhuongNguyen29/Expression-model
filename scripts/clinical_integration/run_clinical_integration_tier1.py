"""
Clinical Integration - Tier 1 Only (No Staging Variables)

Quick test to isolate whether staging variable missingness is causing
the performance decrease in clinical integration.

Tier 1 variables only:
- age: z-scored (1 feature)
- gender: binary (1 feature)
- smoking: Never/Ever/Unknown (3 features)
- alcohol: Never/Ever/Unknown (3 features)

Total: 58 genes + 8 clinical = 66 features

EXCLUDED (due to high missingness in ORIEN):
- N_stage: 26% Unknown in ORIEN vs 4% in TCGA
- T_stage: 37% Unknown in ORIEN vs 3% in TCGA

This script uses existing tuned hyperparameters (no retuning).
If results improve, consider retuning for 66 features.

Usage:
    python run_clinical_integration_tier1.py \
        --data_dir data \
        --consensus_genes consensus_genes.txt \
        --tcga_params hyperparameter_tuning/tcga/best_params.json \
        --orien_params hyperparameter_tuning/orien/best_params.json \
        --output_dir results_v2/08_clinical_integration/tier1_validation

Author: Phuong
Created: 2025
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir
while not (project_root / 'src').exists() and project_root != project_root.parent:
    project_root = project_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset
from src.models.elastic_deepsurv import ElasticDeepSurv
from src.utils.batch_samplers import StratifiedBatchSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1011]

# Tier 1 features (no staging)
TIER1_FEATURES = ['age_zscore', 'gender', 
                  'smoking_Never', 'smoking_Ever', 'smoking_Unknown',
                  'alcohol_Never', 'alcohol_Ever', 'alcohol_Unknown']


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Tier1ClinicalPreprocessor:
    """
    Preprocessor for Tier 1 clinical variables only.
    Excludes N_stage and T_stage due to high missingness in ORIEN.
    """
    
    def __init__(self):
        self.age_mean = None
        self.age_std = None
        self.is_fitted = False
    
    def fit_transform(self, clinical_df: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
        """Fit on source cohort and transform."""
        logger.info(f"Tier1ClinicalPreprocessor: Fitting on {cohort_name}")
        
        # Handle sampleID column
        if 'sampleID' in clinical_df.columns:
            df = clinical_df.set_index('sampleID').copy()
        else:
            df = clinical_df.copy()
        
        # Remove Unnamed column if present
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        result = pd.DataFrame(index=df.index)
        
        # Age: z-score (fit)
        self.age_mean = df['age'].mean()
        self.age_std = df['age'].std()
        result['age_zscore'] = (df['age'] - self.age_mean) / self.age_std
        logger.info(f"  age: mean={self.age_mean:.2f}, std={self.age_std:.2f} -> z-scored")
        
        # Gender: keep as binary
        result['gender'] = df['gender'].values
        logger.info(f"  gender: kept as binary (0/1)")
        
        # Smoking: one-hot encode
        smoking_counts = df['smoking'].value_counts().to_dict()
        logger.info(f"  smoking: {smoking_counts}")
        for cat in ['Never', 'Ever', 'Unknown']:
            result[f'smoking_{cat}'] = (df['smoking'] == cat).astype(int)
        
        # Alcohol: one-hot encode
        alcohol_counts = df['alcohol'].value_counts().to_dict()
        logger.info(f"  alcohol: {alcohol_counts}")
        for cat in ['Never', 'Ever', 'Unknown']:
            result[f'alcohol_{cat}'] = (df['alcohol'] == cat).astype(int)
        
        # EXCLUDED: N_stage, T_stage
        logger.info(f"  N_stage: EXCLUDED (high missingness)")
        logger.info(f"  T_stage: EXCLUDED (high missingness)")
        
        self.is_fitted = True
        logger.info(f"  Output: {result.shape[1]} features (Tier 1 only)")
        
        return result
    
    def transform(self, clinical_df: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
        """Transform target cohort using source parameters."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        
        logger.info(f"Tier1ClinicalPreprocessor: Transforming {cohort_name}")
        
        # Handle sampleID column
        if 'sampleID' in clinical_df.columns:
            df = clinical_df.set_index('sampleID').copy()
        else:
            df = clinical_df.copy()
        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        result = pd.DataFrame(index=df.index)
        
        # Age: z-score using SOURCE parameters
        result['age_zscore'] = (df['age'] - self.age_mean) / self.age_std
        logger.info(f"  age: transformed using source mean/std")
        
        # Gender
        result['gender'] = df['gender'].values
        
        # Smoking
        for cat in ['Never', 'Ever', 'Unknown']:
            result[f'smoking_{cat}'] = (df['smoking'] == cat).astype(int)
        
        # Alcohol
        for cat in ['Never', 'Ever', 'Unknown']:
            result[f'alcohol_{cat}'] = (df['alcohol'] == cat).astype(int)
        
        logger.info(f"  Output: {result.shape[1]} features (Tier 1 only)")
        
        return result


class ClinicalSurvivalDataset(Dataset):
    """Dataset for combined gene + clinical features."""
    
    def __init__(self, features_df: pd.DataFrame, survival_df: pd.DataFrame):
        common_samples = sorted(list(set(features_df.index) & set(survival_df.index)))
        
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


def load_data(data_dir: Path, consensus_gene_file: Path) -> Dict:
    """Load all required data."""
    logger.info("="*60)
    logger.info("Loading Data")
    logger.info("="*60)
    
    # Load expression
    tcga_expr = pd.read_csv(data_dir / "raw" / "tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr = pd.read_csv(data_dir / "raw" / "orien_batch_corrected.csv", index_col=0)
    
    # Load survival
    tcga_surv = pd.read_csv(data_dir / "processed" / "surv_tcga_harmonized.csv", index_col=0)
    orien_surv = pd.read_csv(data_dir / "processed" / "surv_orien_harmonized.csv", index_col=0)
    
    # Load clinical
    tcga_clinical = pd.read_csv(data_dir / "raw" / "clinical_tcga.csv")
    orien_clinical = pd.read_csv(data_dir / "raw" / "clinical_orien_updated.csv")
    
    # Load consensus genes
    with open(consensus_gene_file, 'r') as f:
        consensus_genes = [line.strip() for line in f if line.strip()]
    
    # Filter to consensus genes
    common_genes = [g for g in consensus_genes if g in tcga_expr.index and g in orien_expr.index]
    tcga_expr = tcga_expr.loc[common_genes]
    orien_expr = orien_expr.loc[common_genes]
    
    logger.info(f"  TCGA: {tcga_expr.shape[1]} samples, {len(common_genes)} genes")
    logger.info(f"  ORIEN: {orien_expr.shape[1]} samples, {len(common_genes)} genes")
    
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
        return json.load(f)


def parse_architecture(best_params: Dict) -> List[int]:
    """Parse hidden layer sizes from hyperparameter dict."""
    if 'architecture_2layer' in best_params:
        return [int(x) for x in best_params['architecture_2layer'].split('-')]
    elif 'layer1_size' in best_params:
        return [best_params['layer1_size']]
    else:
        return [96]  # Default


def preprocess_cohort_pair(
    source_expr: pd.DataFrame,
    target_expr: pd.DataFrame,
    source_clinical: pd.DataFrame,
    target_clinical: pd.DataFrame,
    source_name: str,
    target_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Preprocess gene expression and Tier 1 clinical features.
    Fit on source, transform target.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Preprocessing: {source_name} (source) → {target_name} (target)")
    logger.info(f"{'='*60}")
    
    # Gene expression: standardize (fit on source)
    logger.info("\n--- Gene Expression ---")
    gene_scaler = StandardScaler()
    
    source_expr_T = source_expr.T  # samples × genes
    target_expr_T = target_expr.T
    
    source_expr_scaled = gene_scaler.fit_transform(source_expr_T)
    target_expr_scaled = gene_scaler.transform(target_expr_T)
    
    # Clip outliers
    source_expr_scaled = np.clip(source_expr_scaled, -3.0, 3.0)
    target_expr_scaled = np.clip(target_expr_scaled, -3.0, 3.0)
    
    source_expr_df = pd.DataFrame(
        source_expr_scaled,
        index=source_expr.columns,
        columns=source_expr.index
    )
    target_expr_df = pd.DataFrame(
        target_expr_scaled,
        index=target_expr.columns,
        columns=target_expr.index
    )
    
    logger.info(f"  Source: {source_expr_df.shape}")
    logger.info(f"  Target: {target_expr_df.shape}")
    
    # Clinical: Tier 1 only (fit on source)
    logger.info("\n--- Clinical (Tier 1 Only) ---")
    clinical_preprocessor = Tier1ClinicalPreprocessor()
    
    source_clinical_processed = clinical_preprocessor.fit_transform(source_clinical, source_name)
    target_clinical_processed = clinical_preprocessor.transform(target_clinical, target_name)
    
    # Combine features
    logger.info("\n--- Feature Integration ---")
    
    # Align samples
    source_samples = sorted(list(set(source_expr_df.index) & set(source_clinical_processed.index)))
    target_samples = sorted(list(set(target_expr_df.index) & set(target_clinical_processed.index)))
    
    source_combined = pd.concat([
        source_expr_df.loc[source_samples],
        source_clinical_processed.loc[source_samples]
    ], axis=1)
    
    target_combined = pd.concat([
        target_expr_df.loc[target_samples],
        target_clinical_processed.loc[target_samples]
    ], axis=1)
    
    n_genes = len(source_expr.index)
    n_clinical = len(TIER1_FEATURES)
    
    logger.info(f"  Source: {source_combined.shape} ({n_genes} genes + {n_clinical} clinical)")
    logger.info(f"  Target: {target_combined.shape} ({n_genes} genes + {n_clinical} clinical)")
    
    feature_names = list(source_expr.index) + TIER1_FEATURES
    
    return source_combined, target_combined, feature_names


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
    
    return concordance_index(all_times, -np.array(all_risks), all_events)


def train_and_evaluate_single_seed(
    source_cohort: str,
    target_cohort: str,
    data_dict: Dict,
    hyperparams: Dict,
    seed: int,
    epochs: int,
    device: str,
    output_dir: Path
) -> Dict:
    """Train on source, evaluate on target for single seed."""
    set_seed(seed)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Seed {seed}: {source_cohort.upper()} → {target_cohort.upper()}")
    logger.info(f"{'='*60}")
    
    # Get data
    if source_cohort == 'orien':
        source_expr = data_dict['orien_expr']
        source_clinical = data_dict['orien_clinical']
        source_surv = data_dict['orien_surv']
        target_expr = data_dict['tcga_expr']
        target_clinical = data_dict['tcga_clinical']
        target_surv = data_dict['tcga_surv']
    else:
        source_expr = data_dict['tcga_expr']
        source_clinical = data_dict['tcga_clinical']
        source_surv = data_dict['tcga_surv']
        target_expr = data_dict['orien_expr']
        target_clinical = data_dict['orien_clinical']
        target_surv = data_dict['orien_surv']
    
    # Preprocess
    source_features, target_features, feature_names = preprocess_cohort_pair(
        source_expr, target_expr,
        source_clinical, target_clinical,
        source_cohort, target_cohort
    )
    
    # Create datasets
    source_dataset = ClinicalSurvivalDataset(source_features, source_surv)
    target_dataset = ClinicalSurvivalDataset(target_features, target_surv)
    
    # Get hyperparameters
    best_params = hyperparams['best_params']
    hidden_sizes = parse_architecture(best_params)
    
    n_features = source_dataset.n_features
    n_genes = len(data_dict['consensus_genes'])
    n_clinical = len(TIER1_FEATURES)
    
    logger.info(f"\nFeature summary: {n_genes} genes + {n_clinical} clinical = {n_features} total")
    logger.info(f"Architecture: {n_features} → {hidden_sizes} → 1")
    
    # Create dataloaders
    source_sampler = StratifiedBatchSampler(
        events=source_dataset.y_event,
        batch_size=best_params.get('batch_size', 64),
        min_events_per_batch=2,
        shuffle=True
    )
    source_loader = DataLoader(source_dataset, batch_sampler=source_sampler, num_workers=0)
    target_loader = DataLoader(target_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Create model
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=best_params.get('dropout', 0.4),
        activation=best_params.get('activation', 'elu'),
        batch_norm=best_params.get('batch_norm', True),
        weight_init='kaiming_normal',
        l1_ratio=best_params.get('l1_ratio', 0.7),
        alpha=best_params.get('alpha', 0.005)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params.get('learning_rate', 0.0004))
    
    # Train
    logger.info(f"\nTraining for {epochs} epochs...")
    
    training_history = []
    best_test_cindex = 0.0
    best_test_epoch = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, source_loader, optimizer, device)
        
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
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
            
            logger.info(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, "
                       f"Train={train_cindex:.4f}, Test={test_cindex:.4f}")
    
    final_train_cindex = evaluate_model(model, source_loader, device)
    final_test_cindex = evaluate_model(model, target_loader, device)
    
    logger.info(f"\nFinal Results:")
    logger.info(f"  Train C-index: {final_train_cindex:.4f}")
    logger.info(f"  Test C-index: {final_test_cindex:.4f}")
    logger.info(f"  Best Test C-index: {best_test_cindex:.4f} at epoch {best_test_epoch}")
    
    return {
        'seed': seed,
        'source': source_cohort,
        'target': target_cohort,
        'train_cindex': final_train_cindex,
        'test_cindex': final_test_cindex,
        'best_test_cindex': best_test_cindex,
        'best_test_epoch': best_test_epoch,
        'n_features': n_features,
        'n_genes': n_genes,
        'n_clinical': n_clinical,
        'training_history': training_history
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Integration - Tier 1 Only')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--consensus_genes', type=str, required=True)
    parser.add_argument('--tcga_params', type=str, required=True)
    parser.add_argument('--orien_params', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_v2/08_clinical_integration/tier1_validation')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("CLINICAL INTEGRATION - TIER 1 ONLY (NO STAGING)")
    logger.info("="*70)
    logger.info(f"Variables: age, gender, smoking, alcohol")
    logger.info(f"Excluded: N_stage, T_stage (high missingness in ORIEN)")
    logger.info(f"Expected features: 58 genes + 8 clinical = 66 total")
    logger.info("="*70)
    
    # Load data
    data_dict = load_data(data_dir, Path(args.consensus_genes))
    
    # Load hyperparameters
    hyperparams_tcga = load_hyperparameters(Path(args.tcga_params))
    hyperparams_orien = load_hyperparameters(Path(args.orien_params))
    
    # Direction 1: ORIEN → TCGA
    logger.info("\n" + "="*70)
    logger.info("Direction 1: ORIEN → TCGA")
    logger.info("="*70)
    
    orien_to_tcga_results = []
    for seed in args.seeds:
        result = train_and_evaluate_single_seed(
            source_cohort='orien',
            target_cohort='tcga',
            data_dict=data_dict,
            hyperparams=hyperparams_orien,
            seed=seed,
            epochs=args.epochs,
            device=device,
            output_dir=output_dir
        )
        orien_to_tcga_results.append(result)
    
    # Direction 2: TCGA → ORIEN
    logger.info("\n" + "="*70)
    logger.info("Direction 2: TCGA → ORIEN")
    logger.info("="*70)
    
    tcga_to_orien_results = []
    for seed in args.seeds:
        result = train_and_evaluate_single_seed(
            source_cohort='tcga',
            target_cohort='orien',
            data_dict=data_dict,
            hyperparams=hyperparams_tcga,
            seed=seed,
            epochs=args.epochs,
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
    
    # Print results
    logger.info("\n" + "="*70)
    logger.info("TIER 1 RESULTS SUMMARY")
    logger.info("="*70)
    
    n_genes = orien_to_tcga_results[0]['n_genes']
    n_clinical = orien_to_tcga_results[0]['n_clinical']
    n_features = orien_to_tcga_results[0]['n_features']
    
    logger.info(f"\nTier 1 Clinical Integration:")
    logger.info(f"  Features: {n_genes} genes + {n_clinical} clinical = {n_features} total")
    logger.info(f"  Clinical vars: age, gender, smoking, alcohol")
    logger.info(f"  Excluded: N_stage, T_stage")
    logger.info(f"\n  ORIEN → TCGA: {o2t_mean:.4f} ± {o2t_std:.4f}")
    logger.info(f"  TCGA → ORIEN: {t2o_mean:.4f} ± {t2o_std:.4f}")
    logger.info(f"  Mean Bidirectional: {mean_bidirectional:.4f} ± {mean_bidirectional_std:.4f}")
    
    # Comparison
    logger.info(f"\n--- COMPARISON ---")
    logger.info(f"{'Model':<25} | {'O→T':<12} | {'T→O':<12} | {'Mean Bidir':<12}")
    logger.info("-"*65)
    logger.info(f"{'Gene-only (baseline)':<25} | {'0.6803':<12} | {'0.6231':<12} | {'0.6517':<12}")
    logger.info(f"{'Gene+Clinical (all)':<25} | {'0.6677':<12} | {'0.6198':<12} | {'0.6437':<12}")
    logger.info(f"{'Gene+Clinical (Tier1)':<25} | {o2t_mean:<12.4f} | {t2o_mean:<12.4f} | {mean_bidirectional:<12.4f}")
    
    improvement_vs_all = mean_bidirectional - 0.6437
    improvement_vs_baseline = mean_bidirectional - 0.6517
    
    logger.info(f"\n  vs Full Clinical: {improvement_vs_all:+.4f}")
    logger.info(f"  vs Gene-only:     {improvement_vs_baseline:+.4f}")
    
    # Save results
    summary = {
        'experiment': 'tier1_clinical_integration',
        'excluded_variables': ['N_stage', 'T_stage'],
        'included_variables': ['age', 'gender', 'smoking', 'alcohol'],
        'n_features': n_features,
        'n_genes': n_genes,
        'n_clinical': n_clinical,
        'n_seeds': len(args.seeds),
        'epochs': args.epochs,
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
        'baselines': {
            'gene_only': 0.6517,
            'gene_clinical_all': 0.6437
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'tier1_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("="*70)
    
    return summary


if __name__ == '__main__':
    main()
