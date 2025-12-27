"""
Hyperparameter Tuning for Clinical Integration Model

This script tunes hyperparameters for the combined gene + clinical model (73 features):
- 58 consensus genes (from k=90 sign-filtered IG pipeline)
- 15 clinical features (one-hot encoded)

Methodology:
- Leakage-free CV: fit preprocessors on train fold only, transform val fold
- EPV-aware architecture: constraints based on events per variable
- Optuna TPE sampler with median pruning
- Saves CV-derived best epochs for fair comparison with Cox (Option 2)

Architecture search space adjusted for 73 features:
- TCGA (153 events, 73 features): 1-layer, conservative sizes
- ORIEN (450 events, 73 features): 1-2 layers, more flexibility

References:
- Harrell (2001) Regression Modeling Strategies - EPV guidelines
- van der Ploeg et al. (2014) BMC Med Res Methodol - EPV in prediction
- Bergstra & Bengio (2012) JMLR - Hyperparameter optimization

Usage:
    python tune_clinical_integration.py \
        --data_dir data \
        --consensus_genes results_v2/.../consensus_genes.txt \
        --output_dir results_v2/08_clinical_integration/hyperparameter_tuning \
        --n_trials 50

Author: Phuong
Created: 2025
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
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
import yaml
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedKFold
import optuna

from src.data.preprocessor import GeneExpressionPreprocessor
from src.data.clinical_preprocessor import ClinicalPreprocessor
from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_survival_stratification_bins(
    times: np.ndarray,
    events: np.ndarray,
    n_time_bins: int = 4
) -> np.ndarray:
    """
    Create stratification bins combining event status and survival time.
    Based on: Simon et al. (2011), Mogensen et al. (2012)
    """
    strat_bins = np.zeros(len(times), dtype=int)
    
    # Bin censored samples
    censored_mask = (events == 0)
    if censored_mask.sum() > n_time_bins:
        try:
            censored_bins = pd.qcut(
                times[censored_mask],
                q=n_time_bins,
                labels=False,
                duplicates='drop'
            )
            strat_bins[censored_mask] = censored_bins
        except ValueError:
            strat_bins[censored_mask] = 0
    else:
        strat_bins[censored_mask] = 0
    
    # Bin event samples
    event_mask = (events == 1)
    if event_mask.sum() > n_time_bins:
        try:
            event_bins = pd.qcut(
                times[event_mask],
                q=n_time_bins,
                labels=False,
                duplicates='drop'
            )
            strat_bins[event_mask] = event_bins + n_time_bins
        except ValueError:
            strat_bins[event_mask] = n_time_bins
    else:
        strat_bins[event_mask] = n_time_bins
    
    return strat_bins


class ClinicalSurvivalDataset(torch.utils.data.Dataset):
    """
    Dataset for combined gene + clinical features.
    Takes samples × features format directly.
    """
    
    def __init__(self, features_df: pd.DataFrame, survival_df: pd.DataFrame):
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
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': torch.from_numpy(self.X[idx]),
            'time': torch.tensor(self.y_time[idx]),
            'event': torch.tensor(self.y_event[idx]),
            'sample_id': self.sample_ids[idx]
        }


class ClinicalIntegrationTuner:
    """
    Hyperparameter tuner for clinical integration model.
    
    Handles leakage-free CV preprocessing for both gene expression
    and clinical features.
    """
    
    def __init__(
        self,
        expr_raw: pd.DataFrame,
        clinical_df: pd.DataFrame,
        surv_df: pd.DataFrame,
        cohort_name: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        seed: int = 42
    ):
        """
        Initialize tuner.
        
        Args:
            expr_raw: Gene expression DataFrame (genes × samples)
            clinical_df: Clinical DataFrame with sampleID column
            surv_df: Survival DataFrame with time/event columns
            cohort_name: 'tcga' or 'orien'
            device: Training device
            seed: Random seed
        """
        self.expr_raw = expr_raw
        self.clinical_df = clinical_df
        self.surv_df = surv_df
        self.cohort_name = cohort_name
        self.device = device
        self.seed = seed
        
        # Get sample alignment
        expr_samples = set(expr_raw.columns)
        
        # Handle clinical sampleID
        if 'sampleID' in clinical_df.columns:
            clinical_samples = set(clinical_df['sampleID'])
        else:
            clinical_samples = set(clinical_df.index)
        
        surv_samples = set(surv_df.index)
        
        self.common_samples = sorted(list(expr_samples & clinical_samples & surv_samples))
        self.n_samples = len(self.common_samples)
        
        # Get dimensions
        self.n_genes = expr_raw.shape[0]
        self.n_clinical = 15  # Fixed: 1 age + 1 gender + 3 smoking + 3 alcohol + 4 N_stage + 3 T_stage
        self.n_features = self.n_genes + self.n_clinical
        
        # For stratification
        aligned_surv = surv_df.loc[self.common_samples]
        self.events = aligned_surv['event'].values
        self.times = aligned_surv['time'].values
        
        self.strat_bins = create_survival_stratification_bins(
            self.times, self.events, n_time_bins=4
        )
        
        self.n_folds = 5
        self.n_events = int(self.events.sum())
        
        logger.info(f"ClinicalIntegrationTuner initialized")
        logger.info(f"  Cohort: {cohort_name}")
        logger.info(f"  Samples: {self.n_samples}")
        logger.info(f"  Features: {self.n_genes} genes + {self.n_clinical} clinical = {self.n_features} total")
        logger.info(f"  Events: {self.n_events} ({self.events.mean():.1%})")
        logger.info(f"  EPV (events per variable): {self.n_events / self.n_features:.1f}")
        
        self._set_seed(seed)
    
    def _set_seed(self, seed: int):
        """Set all random seeds."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def preprocess_fold(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray
    ) -> Tuple[ClinicalSurvivalDataset, ClinicalSurvivalDataset, int, np.ndarray]:
        """
        Preprocess one CV fold with proper train/val separation.
        
        CRITICAL: Fit preprocessors ONLY on training fold.
        
        Returns:
            train_dataset, val_dataset, n_features, train_events
        """
        # Get sample names for this fold
        train_samples = [self.common_samples[i] for i in train_indices]
        val_samples = [self.common_samples[i] for i in val_indices]
        
        # === Gene Expression Preprocessing ===
        train_expr = self.expr_raw[train_samples]
        val_expr = self.expr_raw[val_samples]
        
        # Standardize genes (fit on train, transform val)
        from sklearn.preprocessing import StandardScaler
        gene_scaler = StandardScaler()
        
        train_expr_T = train_expr.T  # samples × genes
        val_expr_T = val_expr.T
        
        train_expr_scaled = gene_scaler.fit_transform(train_expr_T)
        val_expr_scaled = gene_scaler.transform(val_expr_T)
        
        # Clip outliers
        train_expr_scaled = np.clip(train_expr_scaled, -3.0, 3.0)
        val_expr_scaled = np.clip(val_expr_scaled, -3.0, 3.0)
        
        train_expr_df = pd.DataFrame(
            train_expr_scaled,
            index=train_samples,
            columns=self.expr_raw.index
        )
        val_expr_df = pd.DataFrame(
            val_expr_scaled,
            index=val_samples,
            columns=self.expr_raw.index
        )
        
        # === Clinical Preprocessing ===
        # Get clinical data for fold samples
        if 'sampleID' in self.clinical_df.columns:
            clinical_indexed = self.clinical_df.set_index('sampleID')
        else:
            clinical_indexed = self.clinical_df
        
        # Remove extra columns if present
        if 'Unnamed: 0' in clinical_indexed.columns:
            clinical_indexed = clinical_indexed.drop(columns=['Unnamed: 0'])
        
        train_clinical = clinical_indexed.loc[train_samples]
        val_clinical = clinical_indexed.loc[val_samples]
        
        # Create clinical preprocessor (fit on train, transform val)
        clinical_preprocessor = ClinicalPreprocessor()
        
        # Need to reset index to have sampleID as column for preprocessor
        train_clinical_reset = train_clinical.reset_index()
        train_clinical_reset.columns = ['sampleID'] + list(train_clinical.columns)
        
        val_clinical_reset = val_clinical.reset_index()
        val_clinical_reset.columns = ['sampleID'] + list(val_clinical.columns)
        
        train_clinical_processed = clinical_preprocessor.fit_transform(
            train_clinical_reset, f'{self.cohort_name}_train'
        )
        val_clinical_processed = clinical_preprocessor.transform(
            val_clinical_reset, f'{self.cohort_name}_val'
        )
        
        # === Combine Features ===
        # Align indices
        train_combined = pd.concat([train_expr_df, train_clinical_processed], axis=1)
        val_combined = pd.concat([val_expr_df, val_clinical_processed], axis=1)
        
        # Get survival data
        train_surv = self.surv_df.loc[train_samples]
        val_surv = self.surv_df.loc[val_samples]
        
        # Create datasets
        train_dataset = ClinicalSurvivalDataset(train_combined, train_surv)
        val_dataset = ClinicalSurvivalDataset(val_combined, val_surv)
        
        n_features = train_combined.shape[1]
        train_events = train_surv['event'].values
        
        return train_dataset, val_dataset, n_features, train_events
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function with leakage-free CV.
        
        Architecture search space adjusted for 73 features.
        """
        # =================================================================
        # EPV-AWARE ARCHITECTURE SELECTION FOR 73 FEATURES
        # =================================================================
        # TCGA: 153 events, 73 features → EPV = 2.1 (very constrained)
        # ORIEN: 450 events, 73 features → EPV = 6.2 (more flexibility)
        # =================================================================
        
        m = self.n_features  # 73 features
        
        if self.n_samples < 500:  # TCGA
            # Very constrained: 1 layer only, conservative sizes
            # With 73 features and ~153 events, need strong regularization
            layer_options = [48, 64, 96]
            layer1_size = trial.suggest_categorical('layer1_size', layer_options)
            hidden_sizes = [layer1_size]
            
            # Stronger regularization for low EPV
            alpha = trial.suggest_float('alpha', 5e-4, 1e-2, log=True)
            
        else:  # ORIEN
            # More flexibility with 450 events
            # 73 features falls into "larger feature set" category
            n_layers = trial.suggest_int('n_layers', 1, 2)
            
            if n_layers == 1:
                layer_options = [64, 96, 128]
                layer1_size = trial.suggest_categorical('layer1_size', layer_options)
                hidden_sizes = [layer1_size]
            else:
                architecture = trial.suggest_categorical(
                    'architecture_2layer',
                    ['64-32', '96-48', '128-64']
                )
                hidden_sizes = [int(x) for x in architecture.split('-')]
            
            # Moderate regularization
            alpha = trial.suggest_float('alpha', 1e-4, 5e-3, log=True)
        
        # Common hyperparameters
        dropout = trial.suggest_float('dropout', 0.3, 0.6)
        learning_rate = trial.suggest_float('learning_rate', 5e-6, 5e-4, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.3, 0.9)
        batch_size = trial.suggest_categorical('batch_size', [32, 64])
        activation = trial.suggest_categorical('activation', ['relu', 'elu'])
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])
        
        if batch_norm:
            weight_init = 'kaiming_normal'
        else:
            weight_init = trial.suggest_categorical('weight_init', ['xavier_normal', 'kaiming_normal'])
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        cv_scores = []
        cv_best_epochs = []
        
        sample_indices = np.arange(self.n_samples)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(sample_indices, self.strat_bins)):
            try:
                # Preprocess this fold
                train_dataset, val_dataset, n_features, train_events = self.preprocess_fold(
                    train_idx, val_idx
                )
                
                # Create dataloaders
                train_sampler = StratifiedBatchSampler(
                    events=train_events,
                    batch_size=batch_size,
                    min_events_per_batch=2,
                    shuffle=True
                )
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_sampler=train_sampler,
                    num_workers=0
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0
                )
                
                # Create model
                model = ElasticDeepSurv(
                    n_features=n_features,
                    hidden_sizes=hidden_sizes,
                    dropout=dropout,
                    activation=activation,
                    batch_norm=batch_norm,
                    weight_init=weight_init,
                    l1_ratio=l1_ratio,
                    alpha=alpha
                )
                
                # Create trainer
                trainer = ElasticDeepSurvTrainer(
                    model=model,
                    learning_rate=learning_rate,
                    device=self.device
                )
                
                # Train
                history = trainer.fit(
                    train_loader=train_loader,
                    valid_loader=val_loader,
                    n_epochs=200,
                    early_stopping_patience=30,
                    verbose=False
                )
                
                best_cindex = max(history['valid_c_index'])
                cv_scores.append(best_cindex)
                
                # Capture best epoch for Option 2 (CV-derived stopping)
                fold_best_epoch = history.get('best_epoch', len(history['train_loss']))
                cv_best_epochs.append(fold_best_epoch)
                
            except Exception as e:
                logger.warning(f"Fold {fold+1} failed: {e}")
                cv_scores.append(0.5)
                cv_best_epochs.append(50)
                continue
            
            # Report for pruning
            trial.report(np.mean(cv_scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        mean_cindex = np.mean(cv_scores)
        std_cindex = np.std(cv_scores)
        
        # Store CV best epochs
        mean_best_epoch = int(np.mean(cv_best_epochs))
        trial.set_user_attr('cv_best_epochs', cv_best_epochs)
        trial.set_user_attr('mean_best_epoch', mean_best_epoch)
        
        # Add sparsity penalty
        try:
            sparsity_info = model.get_sparsity_info()
            sparsity_ratio = sparsity_info['sparsity_ratio']
            
            MIN_SPARSITY = 0.05
            if sparsity_ratio < MIN_SPARSITY:
                sparsity_penalty = 0.1 * (MIN_SPARSITY - sparsity_ratio)
            else:
                sparsity_penalty = 0.0
            
            objective_value = mean_cindex - sparsity_penalty
        except:
            objective_value = mean_cindex
        
        logger.info(f"Trial {trial.number}: C-index={mean_cindex:.4f} ± {std_cindex:.4f}, "
                   f"Arch={hidden_sizes}, Alpha={alpha:.4f}")
        
        return objective_value
    
    def optimize(self, n_trials: int = 50) -> Tuple[Dict, optuna.Study, Dict]:
        """
        Run hyperparameter optimization.
        
        Returns:
            best_params, study, cv_epochs_info
        """
        study_name = f"clinical_integration_{self.cohort_name}"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"HYPERPARAMETER OPTIMIZATION: {self.cohort_name.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Features: {self.n_genes} genes + {self.n_clinical} clinical = {self.n_features} total")
        logger.info(f"Samples: {self.n_samples}, Events: {self.n_events}")
        logger.info(f"EPV: {self.n_events / self.n_features:.1f}")
        logger.info(f"CV: {self.n_folds}-fold stratified")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"{'='*60}\n")
        
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        )
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # Extract CV-derived epochs from best trial
        best_trial = study.best_trial
        cv_best_epochs = best_trial.user_attrs.get('cv_best_epochs', [100] * self.n_folds)
        mean_best_epoch = best_trial.user_attrs.get('mean_best_epoch', 100)
        
        cv_epochs_info = {
            'cv_best_epochs': cv_best_epochs,
            'mean_best_epoch': mean_best_epoch,
            'std_best_epoch': float(np.std(cv_best_epochs)),
            'min_best_epoch': min(cv_best_epochs),
            'max_best_epoch': max(cv_best_epochs)
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Best CV C-index: {study.best_value:.4f}")
        logger.info(f"Best parameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"\nCV-derived epochs:")
        logger.info(f"  Per-fold: {cv_best_epochs}")
        logger.info(f"  Mean: {mean_best_epoch} ± {cv_epochs_info['std_best_epoch']:.1f}")
        logger.info(f"{'='*60}\n")
        
        return study.best_params, study, cv_epochs_info


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
    
    logger.info(f"  TCGA: {tcga_expr.shape[1]} samples")
    logger.info(f"  ORIEN: {orien_expr.shape[1]} samples")
    logger.info(f"  Consensus genes: {len(consensus_genes)}")
    
    # Filter to consensus genes
    common_genes = [g for g in consensus_genes if g in tcga_expr.index and g in orien_expr.index]
    tcga_expr = tcga_expr.loc[common_genes]
    orien_expr = orien_expr.loc[common_genes]
    
    logger.info(f"  Using {len(common_genes)} genes")
    
    return {
        'tcga_expr': tcga_expr,
        'orien_expr': orien_expr,
        'tcga_surv': tcga_surv,
        'orien_surv': orien_surv,
        'tcga_clinical': tcga_clinical,
        'orien_clinical': orien_clinical,
        'consensus_genes': common_genes
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Clinical Integration')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--consensus_genes', type=str, required=True,
                        help='Path to consensus genes file')
    parser.add_argument('--output_dir', type=str, 
                        default='results_v2/08_clinical_integration/hyperparameter_tuning',
                        help='Output directory')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials per cohort')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')
    parser.add_argument('--cohort', type=str, default='both',
                        choices=['tcga', 'orien', 'both'],
                        help='Which cohort(s) to tune')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("HYPERPARAMETER TUNING FOR CLINICAL INTEGRATION")
    logger.info("="*70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Consensus genes: {args.consensus_genes}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Trials per cohort: {args.n_trials}")
    logger.info(f"Device: {device}")
    logger.info(f"Cohort(s): {args.cohort}")
    logger.info("="*70)
    
    # Load data
    data = load_data(data_dir, Path(args.consensus_genes))
    
    results = {}
    
    # Tune TCGA
    if args.cohort in ['tcga', 'both']:
        logger.info("\n" + "="*70)
        logger.info("TUNING TCGA")
        logger.info("="*70)
        
        tcga_tuner = ClinicalIntegrationTuner(
            expr_raw=data['tcga_expr'],
            clinical_df=data['tcga_clinical'],
            surv_df=data['tcga_surv'],
            cohort_name='tcga',
            device=device
        )
        
        tcga_params, tcga_study, tcga_epochs = tcga_tuner.optimize(n_trials=args.n_trials)
        
        # Save TCGA results
        tcga_dir = output_dir / 'tcga'
        tcga_dir.mkdir(parents=True, exist_ok=True)
        
        tcga_results = {
            'cohort': 'tcga',
            'n_features': tcga_tuner.n_features,
            'n_genes': tcga_tuner.n_genes,
            'n_clinical': tcga_tuner.n_clinical,
            'n_samples': tcga_tuner.n_samples,
            'n_events': tcga_tuner.n_events,
            'best_params': tcga_params,
            'best_cv_cindex': tcga_study.best_value,
            'n_trials': args.n_trials,
            'cv_epochs_info': tcga_epochs,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(tcga_dir / 'best_params.json', 'w') as f:
            json.dump(tcga_results, f, indent=2)
        
        tcga_study.trials_dataframe().to_csv(tcga_dir / 'trials.csv', index=False)
        
        results['tcga'] = tcga_results
        logger.info(f"TCGA results saved to: {tcga_dir}")
    
    # Tune ORIEN
    if args.cohort in ['orien', 'both']:
        logger.info("\n" + "="*70)
        logger.info("TUNING ORIEN")
        logger.info("="*70)
        
        orien_tuner = ClinicalIntegrationTuner(
            expr_raw=data['orien_expr'],
            clinical_df=data['orien_clinical'],
            surv_df=data['orien_surv'],
            cohort_name='orien',
            device=device
        )
        
        orien_params, orien_study, orien_epochs = orien_tuner.optimize(n_trials=args.n_trials)
        
        # Save ORIEN results
        orien_dir = output_dir / 'orien'
        orien_dir.mkdir(parents=True, exist_ok=True)
        
        orien_results = {
            'cohort': 'orien',
            'n_features': orien_tuner.n_features,
            'n_genes': orien_tuner.n_genes,
            'n_clinical': orien_tuner.n_clinical,
            'n_samples': orien_tuner.n_samples,
            'n_events': orien_tuner.n_events,
            'best_params': orien_params,
            'best_cv_cindex': orien_study.best_value,
            'n_trials': args.n_trials,
            'cv_epochs_info': orien_epochs,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(orien_dir / 'best_params.json', 'w') as f:
            json.dump(orien_results, f, indent=2)
        
        orien_study.trials_dataframe().to_csv(orien_dir / 'trials.csv', index=False)
        
        results['orien'] = orien_results
        logger.info(f"ORIEN results saved to: {orien_dir}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TUNING SUMMARY")
    logger.info("="*70)
    
    for cohort, res in results.items():
        logger.info(f"\n{cohort.upper()}:")
        logger.info(f"  Features: {res['n_genes']} genes + {res['n_clinical']} clinical = {res['n_features']} total")
        logger.info(f"  Best CV C-index: {res['best_cv_cindex']:.4f}")
        logger.info(f"  Best params: {res['best_params']}")
        logger.info(f"  CV epochs: {res['cv_epochs_info']['mean_best_epoch']} ± {res['cv_epochs_info']['std_best_epoch']:.1f}")
    
    logger.info("\n" + "="*70)
    logger.info("NEXT STEP: Run cross-cohort validation with tuned parameters")
    logger.info(f"  Use: python run_clinical_integration.py --tcga_params {output_dir}/tcga/best_params.json --orien_params {output_dir}/orien/best_params.json")
    logger.info("="*70)
    
    return results


if __name__ == '__main__':
    main()
