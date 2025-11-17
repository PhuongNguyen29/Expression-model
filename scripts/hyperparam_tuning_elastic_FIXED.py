#!/usr/bin/env python3
"""
Hyperparameter tuning with proper cross-validation (no data leakage) and gradient fixes.

CRITICAL FIXES IMPLEMENTED:
1. Xavier initialization DISABLED when BatchNorm=True (incompatible, causes gradient issues)
2. Adaptive gradient thresholds based on parameter count
3. Proper CV with independent preprocessing per fold

Evidence Base:
- Simon et al., 2003: "Pitfalls in DNA microarray data analysis"
- Ambroise & McLachlan, 2002: "Selection bias in gene extraction"
- Glorot & Bengio, 2010: Xavier initialization
- He et al., 2015: Kaiming initialization for ReLU networks

Author: Phuong
Date: 2024-11-17
Status: ACTIVE - Step 1 of transfer learning pipeline
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

import torch
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
import optuna
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import yaml
import json
import pickle

from src.data.preprocessor import GeneExpressionPreprocessor
from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    
    This ensures each CV fold has:
    1. Similar proportion of events vs censored
    2. Similar distribution of survival times
    """
    import pandas as pd
    
    strat_bins = np.zeros(len(times), dtype=int)
    
    # Bin censored samples by time quartiles
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
    
    # Bin event samples by time quartiles (offset by n_time_bins)
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
    
    logger.info(f"Created {len(np.unique(strat_bins))} stratification bins")
    
    return strat_bins


class LeakageFreeHyperparameterTuner:
    """
    Hyperparameter tuner with proper CV preprocessing and gradient fixes.
    
    Key Features:
    1. Proper CV: Fit preprocessor only on training folds
    2. Gradient-safe: Xavier disabled with BatchNorm
    3. Stratification: Balanced event/time distribution
    """
    
    def __init__(
        self,
        config: dict,
        cohort_name: str,
        n_folds: int = 5,
        seed: int = 42
    ):
        """
        Initialize tuner with configuration.
        
        Args:
            config: Configuration dictionary
            cohort_name: 'tcga' or 'orien'
            n_folds: Number of CV folds
            seed: Random seed
        """
        self.config = config
        self.cohort_name = cohort_name
        self.n_folds = n_folds
        self.seed = seed
        
        # Load RAW data (not preprocessed - critical for proper CV)
        self._load_raw_data()
        
        # Set random seeds
        self._set_seeds(seed)
        
        # Create stratification bins for survival analysis
        self.strat_bins = create_survival_stratification_bins(
            self.times,
            self.events
        )
        
    def _load_raw_data(self):
        """Load RAW batch-corrected data (before IQR filtering)."""
        data_config = self.config['data']
        raw_dir = Path(data_config['raw_data_dir'])
        
        logger.info(f"Loading RAW data for {self.cohort_name}...")
        
        # Load expression data
        if self.cohort_name == 'tcga':
            expr_file = raw_dir / data_config['tcga_expression']
            surv_file = raw_dir / data_config['tcga_survival']
        else:  # orien
            expr_file = raw_dir / data_config['orien_expression']
            surv_file = raw_dir / data_config['orien_survival']
        
        self.train_expr_raw = pd.read_csv(expr_file, index_col=0)
        self.train_surv = pd.read_csv(surv_file, index_col=0)
        
        # Align samples
        common_samples = self.train_expr_raw.columns.intersection(self.train_surv.index)
        self.train_expr_raw = self.train_expr_raw[common_samples]
        self.train_surv = self.train_surv.loc[common_samples]
        
        # Filter to 308 consensus genes
        consensus_file = raw_dir / 'consensus_genes_308.txt'
        with open(consensus_file, 'r') as f:
            consensus_genes = [line.strip() for line in f if line.strip()]
        
        # Keep only consensus genes present in this cohort
        available_genes = [g for g in consensus_genes if g in self.train_expr_raw.index]
        self.train_expr_raw = self.train_expr_raw.loc[available_genes]
        
        self.n_samples = len(common_samples)
        self.n_genes_raw = len(available_genes)
        self.times = self.train_surv['time'].values
        self.events = self.train_surv['event'].values
        
        logger.info(f"  Samples: {self.n_samples}")
        logger.info(f"  Genes (308 consensus): {self.n_genes_raw}")
        logger.info(f"  Events: {self.events.sum()} ({self.events.mean():.1%})")
        logger.info(f"  Median survival: {np.median(self.times):.0f} days")
    
    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
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
    ):
        """
        Preprocess one CV fold with proper train/test separation.
        
        Returns:
            train_dataset, val_dataset, n_features, train_events
        """
        # Get sample names for this fold
        train_samples = self.train_expr_raw.columns[train_indices]
        val_samples = self.train_expr_raw.columns[val_indices]
        
        # Extract raw data for this fold
        train_fold_raw = self.train_expr_raw[train_samples]
        val_fold_raw = self.train_expr_raw[val_samples]
        
        # Create preprocessor for this fold
        fold_preprocessor = GeneExpressionPreprocessor(self.config)
        
        # CRITICAL: Fit preprocessor ONLY on training fold
        train_processed = fold_preprocessor.fit_transform_single_cohort(
            train_fold_raw,
            cohort_name=f"{self.cohort_name}_train"
        )
        
        # Transform validation fold using training parameters
        val_processed = fold_preprocessor.transform_single_cohort(val_fold_raw)
        
        # Get survival data for this fold
        train_surv_fold = self.train_surv.loc[train_samples]
        val_surv_fold = self.train_surv.loc[val_samples]
        
        # Create datasets
        train_dataset = SurvivalDataset(train_processed, train_surv_fold)
        val_dataset = SurvivalDataset(val_processed, val_surv_fold)
        
        n_features = train_processed.shape[0]
        
        # Return events for stratified batch sampling
        train_events = train_surv_fold['event'].values
        
        return train_dataset, val_dataset, n_features, train_events
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective with proper CV preprocessing and gradient-safe hyperparameters.
        
        Returns:
            Mean C-index across folds
        """
        # Sample hyperparameters based on cohort size
        if self.n_samples < 500:  # TCGA (small cohort)
            n_layers = trial.suggest_int('n_layers', 1, 2)
            if n_layers == 1:
                layer1_size = trial.suggest_categorical('layer1_size', [64, 128, 256])
                hidden_sizes = [layer1_size]
                alpha = trial.suggest_float('alpha', 5e-5, 1e-3, log=True)
            else:  # n_layers == 2
                architecture = trial.suggest_categorical(
                    'architecture_2layer',
                    ['256-64', '256-128', '128-64', '128-32']
                )
                hidden_sizes = [int(x) for x in architecture.split('-')]
                alpha = trial.suggest_float('alpha', 5e-5, 1e-3, log=True)
            dropout = trial.suggest_categorical('dropout', [0.2, 0.3, 0.4])
            batch_size = trial.suggest_categorical('batch_size', [32, 48])
            
        else:  # ORIEN (large cohort)
            n_layers = trial.suggest_int('n_layers', 2, 3)
            if n_layers == 2:
                architecture = trial.suggest_categorical(
                    'architecture_2layer',
                    ['256-128', '256-64', '128-64', '192-96']
                )
                hidden_sizes = [int(x) for x in architecture.split('-')]
            else:  # n_layers == 3
                architecture = trial.suggest_categorical(
                    'architecture_3layer',
                    ['256-128-32', '192-96-32', '128-64-32']
                )
                hidden_sizes = [int(x) for x in architecture.split('-')]
            dropout = trial.suggest_categorical('dropout', [0.3, 0.4, 0.5])
            batch_size = trial.suggest_categorical('batch_size', [32, 64])
            alpha = trial.suggest_float('alpha', 5e-6, 1e-4, log=True)
        
        # Common hyperparameters
        activation = trial.suggest_categorical('activation', ['relu', 'elu'])
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])
        
        # CRITICAL FIX: Prevent Xavier+BatchNorm combination
        # Based on Chapter 4 gradient explosion analysis
        if batch_norm:
            weight_init = 'kaiming_uniform'  # Always use Kaiming with BatchNorm
            logger.debug("  Using kaiming_uniform (BatchNorm enabled)")
        else:
            weight_init = trial.suggest_categorical(
                'weight_init',
                ['xavier_normal', 'kaiming_uniform']
            )
            logger.debug(f"  Using {weight_init} (BatchNorm disabled)")
        
        # Elastic Net parameters
        l1_ratio = trial.suggest_categorical('l1_ratio', [0.3, 0.5, 0.7, 0.9])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        
        # Cross-validation
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.seed
        )
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(
            range(self.n_samples),
            self.strat_bins
        )):
            # Log fold statistics
            train_event_rate = self.events[train_idx].mean()
            val_event_rate = self.events[val_idx].mean()
            train_median_time = np.median(self.times[train_idx])
            val_median_time = np.median(self.times[val_idx])
            
            logger.info(f"\n  Trial {trial.number}, Fold {fold+1}/{self.n_folds}:")
            logger.info(f"    Train: {len(train_idx)} samples, "
                       f"{train_event_rate:.1%} events, "
                       f"median time = {train_median_time:.0f}")
            logger.info(f"    Val:   {len(val_idx)} samples, "
                       f"{val_event_rate:.1%} events, "
                       f"median time = {val_median_time:.0f}")
            
            # CRITICAL: Preprocess this fold independently
            train_dataset, val_dataset, n_features, train_events = self.preprocess_fold(
                train_idx, val_idx
            )
            
            logger.info(f"    Features after preprocessing: {n_features}")
            
            # Create stratified batch sampler
            train_sampler = StratifiedBatchSampler(
                events=train_events,
                batch_size=batch_size,
                shuffle=True
            )
            
            # Create data loaders
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
            
            # Build model configuration
            model_config = {
                'model': {
                    'hidden_sizes': hidden_sizes,
                    'dropout': dropout,
                    'activation': activation,
                    'batch_norm': batch_norm,
                    'weight_init': weight_init,
                    'l1_ratio': l1_ratio,
                    'alpha': alpha
                }
            }
            
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
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            trainer = ElasticDeepSurvTrainer(
                model=model,
                learning_rate=learning_rate,
                device=device
            )
            
            # Training configuration
            n_epochs = 100
            patience = 20
            best_c_index = 0.0
            patience_counter = 0
            
            # Train model
            try:
                for epoch in range(n_epochs):
                    # Train epoch
                    train_loss, train_cox, train_penalty = trainer.train_epoch(train_loader)
                    
                    # Validate
                    val_loss, val_c_index, val_cox, val_penalty = trainer.evaluate(val_loader)
                    
                    # Early stopping
                    if val_c_index > best_c_index:
                        best_c_index = val_c_index
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        logger.info(f"    Early stopping at epoch {epoch+1}")
                        break
                
                cv_scores.append(best_c_index)
                logger.info(f"    Fold C-index: {best_c_index:.4f}")
                
            except Exception as e:
                logger.error(f"    Trial failed: {str(e)}")
                return 0.0  # Return poor score for failed trials
            
            # Report intermediate value to Optuna pruner
            trial.report(np.mean(cv_scores), fold)
            
            # Prune if needed
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        mean_c_index = np.mean(cv_scores)
        logger.info(f"\n  Trial {trial.number} completed: "
                   f"Mean C-index = {mean_c_index:.4f} ± {np.std(cv_scores):.4f}")
        
        return mean_c_index
    
    def optimize(self, n_trials: int = 50, study_name: str = None):
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of Optuna trials
            study_name: Name for the study
            
        Returns:
            best_params, study
        """
        if study_name is None:
            study_name = f"elastic_deepsurv_{self.cohort_name}_v2"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING HYPERPARAMETER OPTIMIZATION (V2 - GRADIENT-SAFE)")
        logger.info(f"{'='*60}")
        logger.info(f"Cohort: {self.cohort_name}")
        logger.info(f"Samples: {self.n_samples}")
        logger.info(f"Genes: {self.n_genes_raw}")
        logger.info(f"CV strategy: {self.n_folds}-fold stratified")
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
        
        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Best CV C-index: {study.best_value:.4f}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best parameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4e}")
            else:
                logger.info(f"  {key}: {value}")
        
        return study.best_params, study


def run_tuning(
    cohort: str = 'tcga',
    n_trials: int = 50,
    output_dir: str = None
):
    """
    Run hyperparameter tuning with proper CV preprocessing and gradient fixes.
    
    Args:
        cohort: 'tcga' or 'orien'
        n_trials: Number of Optuna trials
        output_dir: Where to save results
    """
    # Load configuration
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_v2/01_hyperparameter_tuning/{cohort}_308genes_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Create tuner
    tuner = LeakageFreeHyperparameterTuner(
        config=config,
        cohort_name=cohort,
        n_folds=5,
        seed=42
    )
    
    # Run optimization
    best_params, study = tuner.optimize(n_trials=n_trials)
    
    # Save results
    # 1. Best parameters
    with open(Path(output_dir) / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # 2. CV performance
    cv_performance = {
        'mean_c_index': study.best_value,
        'std_c_index': np.std([t.value for t in study.trials if t.value is not None]),
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials)
    }
    with open(Path(output_dir) / 'cv_performance.json', 'w') as f:
        json.dump(cv_performance, f, indent=2)
    
    # 3. All trials
    trials_df = study.trials_dataframe()
    trials_df.to_csv(Path(output_dir) / 'trials.csv', index=False)
    
    # 4. Study object
    with open(Path(output_dir) / 'study.pkl', 'wb') as f:
        pickle.dump(study, f)
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - best_params.json")
    logger.info(f"  - cv_performance.json")
    logger.info(f"  - trials.csv ({len(trials_df)} trials)")
    logger.info(f"  - study.pkl")
    
    return best_params, study


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning with gradient-safe configurations'
    )
    parser.add_argument(
        '--cohort',
        type=str,
        required=True,
        choices=['tcga', 'orien'],
        help='Cohort to tune: tcga or orien'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of Optuna trials (default: 50)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated in results_v2/)'
    )
    
    args = parser.parse_args()
    
    run_tuning(
        cohort=args.cohort,
        n_trials=args.n_trials,
        output_dir=args.output_dir
    )