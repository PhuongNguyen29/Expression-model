"""
Hyperparameter tuning with PROPER cross-validation (no data leakage).

Key changes from original:
1. Load RAW data (not preprocessed)
2. Fit preprocessor inside each CV fold
3. Each fold gets independent preprocessing

Based on:
- Simon et al. (2003), "Pitfalls in DNA microarray data"
- Ambroise & McLachlan (2002), "Selection bias in gene extraction"
"""

import sys
sys.path.append('.')

import torch
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import optuna
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

from src.data.preprocessor import GeneExpressionPreprocessor
from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LeakageFreeHyperparameterTuner:
    """
    Hyperparameter tuner with proper CV preprocessing.
    
    Ensures unbiased performance estimation by:
    1. Fitting preprocessor only on training folds
    2. Transforming test folds with training parameters
    3. No information leakage from test to train
    """
    
    def __init__(
        self,
        train_expr_raw: pd.DataFrame,  # RAW expression data
        train_surv: pd.DataFrame,
        config: dict,
        cohort_name: str = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        seed: int = 42
    ):
        self.train_expr_raw = train_expr_raw  # Keep raw data
        self.train_surv = train_surv
        self.config = config
        self.cohort_name = cohort_name
        self.device = device
        self.seed = seed
        
        self.n_samples = train_expr_raw.shape[1]
        self.n_genes_raw = train_expr_raw.shape[0]
        self.events = train_surv['event'].values
        
        self.n_folds = 5
        
        logger.info(f"LeakageFreeHyperparameterTuner initialized")
        logger.info(f"  Cohort: {cohort_name}")
        logger.info(f"  Samples: {self.n_samples}")
        logger.info(f"  Raw genes: {self.n_genes_raw}")
        logger.info(f"  Event rate: {self.events.mean():.1%}")
        
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
    ):
        """
        Preprocess one CV fold with proper train/test separation.
        
        CRITICAL: This is where we prevent data leakage.
        
        Args:
            train_indices: Sample indices for training
            val_indices: Sample indices for validation
            
        Returns:
            train_dataset, val_dataset, n_features
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
        
        return train_dataset, val_dataset, n_features
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective with proper CV preprocessing.
        
        Returns:
            Mean C-index across folds
        """
        # Sample hyperparameters
        if self.n_samples < 500:  # TCGA
            n_layers = trial.suggest_int('n_layers', 1, 2)
            if n_layers == 1:
                hidden_sizes = [trial.suggest_categorical('h1', [128, 256])]
            else:
                h1 = trial.suggest_categorical('h1', [256, 384])
                h2 = trial.suggest_categorical('h2', [64, 128])
                hidden_sizes = [h1, h2]
            dropout = trial.suggest_categorical('dropout', [0.2, 0.3, 0.4])
            batch_size = trial.suggest_categorical('batch_size', [32, 48])
        else:  # ORIEN
            n_layers = trial.suggest_int('n_layers', 2, 3)
            if n_layers == 2:
                pattern = trial.suggest_categorical('pattern', ['512-128', '384-96', '256-64'])
                hidden_sizes = [int(x) for x in pattern.split('-')]
            else:
                pattern = trial.suggest_categorical('pattern', ['512-256-64', '384-192-48'])
                hidden_sizes = [int(x) for x in pattern.split('-')]
            dropout = trial.suggest_categorical('dropout', [0.3, 0.4, 0.5])
            batch_size = trial.suggest_categorical('batch_size', [32, 64])
        
        activation = trial.suggest_categorical('activation', ['relu', 'elu'])
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])
        weight_init = trial.suggest_categorical('weight_init', ['xavier_normal', 'kaiming_uniform'])
        
        # ElasticDeepSurv-specific parameters
        alpha = trial.suggest_float('alpha', 1e-4, 1e-2, log=True)
        l1_ratio = trial.suggest_categorical('l1_ratio', [0.5, 0.7, 0.9])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        
        # Cross-validation loop
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(
            range(self.n_samples), self.events
        )):
            logger.info(f"\n  Fold {fold+1}/{self.n_folds}")
            
            # CRITICAL: Preprocess this fold independently
            train_dataset, val_dataset, n_features = self.preprocess_fold(
                train_idx, val_idx
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # Create model for this fold
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
                weight_decay=0.0,  # Use elastic net instead
                device=self.device
            )
            
            # Train with early stopping
            try:
                history = trainer.fit(
                    train_loader=train_loader,
                    valid_loader=val_loader,
                    n_epochs=100,
                    early_stopping_patience=15,
                    verbose=False
                )
                
                best_cindex = max(history['valid_c_index'])
                cv_scores.append(best_cindex)
                
                logger.info(f"    C-index: {best_cindex:.4f}")
                
            except Exception as e:
                logger.warning(f"    Fold failed: {e}")
                cv_scores.append(0.5)
                continue
            
            # Report for pruning
            trial.report(np.mean(cv_scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        mean_cindex = np.mean(cv_scores)
        std_cindex = np.std(cv_scores)
        
        logger.info(f"Trial {trial.number}: {mean_cindex:.4f} ± {std_cindex:.4f}")
        
        return mean_cindex
    
    def optimize(self, n_trials: int = 50, study_name: str = None):
        """Run optimization."""
        if study_name is None:
            study_name = f"elastic_deepsurv_{self.cohort_name}_FIXED"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING HYPERPARAMETER OPTIMIZATION (LEAKAGE-FREE)")
        logger.info(f"{'='*60}")
        logger.info(f"Cohort: {self.cohort_name}")
        logger.info(f"Samples: {self.n_samples}")
        logger.info(f"Raw genes: {self.n_genes_raw}")
        logger.info(f"CV strategy: {self.n_folds}-fold stratified (proper preprocessing)")
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


def run_leakage_free_tuning(
    cohort: str = 'tcga',
    n_trials: int = 50,
    output_dir: str = None
):
    """
    Run hyperparameter tuning with proper CV preprocessing.
    
    Args:
        cohort: 'tcga' or 'orien'
        n_trials: Number of Optuna trials
        output_dir: Where to save results
    """
    import yaml
    import json
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/hyperparam_FIXED_{cohort}_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load RAW data (not preprocessed!)
    logger.info(f"Loading RAW data for {cohort}...")
    
    if cohort.lower() == 'tcga':
        expr_raw = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
        surv = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    else:  # orien
        expr_raw = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
        surv = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    logger.info(f"Loaded: {expr_raw.shape[0]} genes × {expr_raw.shape[1]} samples")
    
    # Create tuner
    tuner = LeakageFreeHyperparameterTuner(
        train_expr_raw=expr_raw,
        train_surv=surv,
        config=config,
        cohort_name=cohort
    )
    
    # Run optimization
    best_params, study = tuner.optimize(n_trials=n_trials)
    
    # Save results
    with open(f"{output_dir}/best_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    study.trials_dataframe().to_csv(f"{output_dir}/trials.csv", index=False)
    
    # Save summary
    summary = {
        'cohort': cohort,
        'n_samples': expr_raw.shape[1],
        'n_genes_raw': expr_raw.shape[0],
        'cv_method': '5-fold stratified with per-fold preprocessing',
        'data_leakage': 'NONE (proper CV)',
        'best_cv_cindex': study.best_value,
        'best_params': best_params,
        'n_trials': len(study.trials)
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"{'='*60}\n")
    
    return best_params, study


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='tcga', choices=['tcga', 'orien'])
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    best_params, study = run_leakage_free_tuning(
        cohort=args.cohort,
        n_trials=args.n_trials,
        output_dir=args.output_dir
    )