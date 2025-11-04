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
    Hyperparameter tuner with proper CV preprocessing.
    
    Ensures unbiased performance estimation by:
    1. Fitting preprocessor only on training folds
    2. Transforming test folds with training parameters
    3. No information leakage from test to train
    """
    
    def __init__(
    self,
    train_expr_raw: pd.DataFrame,
    train_surv: pd.DataFrame,
    config: dict,
    cohort_name: str = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42
    ):
        self.train_expr_raw = train_expr_raw
        self.train_surv = train_surv
        self.config = config
        self.cohort_name = cohort_name
        self.device = device
        self.seed = seed
        
        self.n_samples = train_expr_raw.shape[1]
        self.n_genes_raw = train_expr_raw.shape[0]
        
        # For stratification
        self.events = train_surv['event'].values
        self.times = train_surv['time'].values
        
        # Create enhanced stratification bins (event + time)
        self.strat_bins = create_survival_stratification_bins(
            self.times,
            self.events,
            n_time_bins=4
        )
        
        self.n_folds = 5
        
        logger.info(f"LeakageFreeHyperparameterTuner initialized")
        logger.info(f"  Cohort: {cohort_name}")
        logger.info(f"  Samples: {self.n_samples}")
        logger.info(f"  Raw genes: {self.n_genes_raw}")
        logger.info(f"  Event rate: {self.events.mean():.1%}")
        logger.info(f"  Median survival: {np.median(self.times):.0f}")
        logger.info(f"  Stratification bins: {len(np.unique(self.strat_bins))}")
        
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
        Optuna objective with proper CV preprocessing.
        
        Returns:
            Mean C-index across folds
        """
        # Sample hyperparameters
        if self.n_samples < 500:  # TCGA
            n_layers = trial.suggest_int('n_layers', 1, 2)
            if n_layers == 1:
                layer1_size = trial.suggest_categorical('layer1_size', [128, 256, 384])
                hidden_sizes = [layer1_size]
            else:  # n_layers == 2
            # Two layers: use predefined patterns to avoid Optuna conflicts
                architecture = trial.suggest_categorical(
                    'architecture_2layer',
                    ['256-64',   # 3.80M params - narrow funnel
                    '256-128',  # 3.82M params - wider second layer
                    '384-64',   # 5.68M params - aggressive funnel
                    '384-128']) # 5.72M params - moderate funnel
                hidden_sizes = [int(x) for x in architecture.split('-')]
            dropout = trial.suggest_categorical('dropout', [0.2, 0.3, 0.4])
            batch_size = trial.suggest_categorical('batch_size', [32, 48])
        else:  # ORIEN
            n_layers = trial.suggest_int('n_layers', 2, 3)
            if n_layers == 2:
                architecture = trial.suggest_categorical(
                    'architecture_2layer',
                    ['512-128',  # 7.60M params
                    '384-96',   # 5.68M params
                    '256-64'])  # 3.80M params
                hidden_sizes = [int(x) for x in architecture.split('-')]
            else:
                architecture = trial.suggest_categorical(
                'architecture_3layer',
                ['512-256-64',   # 7.66M params - gradual funnel
                '384-192-48',   # 5.69M params - proportional reduction
                '256-128-32'])  # 3.82M params - conservative
                hidden_sizes = [int(x) for x in architecture.split('-')]
            dropout = trial.suggest_categorical('dropout', [0.3, 0.4, 0.5])
            batch_size = trial.suggest_categorical('batch_size', [32, 64])
        
        activation = trial.suggest_categorical('activation', ['relu', 'elu'])
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])
        weight_init = trial.suggest_categorical('weight_init', ['xavier_normal', 'kaiming_uniform'])
        
        # ElasticDeepSurv-specific parameters
        alpha = trial.suggest_float('alpha', 1e-4, 1e-2, log=True)
        l1_ratio = trial.suggest_categorical('l1_ratio', [0.5, 0.7, 0.9])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
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
            
            logger.info(f"\n  Fold {fold+1}/{self.n_folds}:")
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
            
            # ============================================================
            # CREATE STRATIFIED BATCH SAMPLER (NEW - CRITICAL FIX)
            # ============================================================
            
            train_batch_sampler = StratifiedBatchSampler(
                events=train_events,
                batch_size=batch_size,
                min_events_per_batch=1,  # Guarantee at least 1 event per batch
                shuffle=True,
                drop_last=False
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler  # Use custom sampler
            )
            
            # Validation doesn't need stratified sampling (no gradient computation)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            logger.info(f"    Train batches: {len(train_loader)}")
            logger.info(f"    Val batches: {len(val_loader)}")
            
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
                weight_decay=0.0,
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
                
                logger.info(f"    Best C-index: {best_cindex:.4f}")
                
            except Exception as e:
                logger.warning(f"    Fold {fold+1} failed: {e}")
                cv_scores.append(0.5)
                continue
            
            # Report for pruning
            trial.report(np.mean(cv_scores), fold)
            if trial.should_prune():
                logger.info(f"  Trial pruned at fold {fold+1}")
                raise optuna.TrialPruned()
        
        mean_cindex = np.mean(cv_scores)
        std_cindex = np.std(cv_scores)
        
        logger.info(f"\nTrial {trial.number}: {mean_cindex:.4f} ± {std_cindex:.4f}")
        
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
    if config['data'].get('use_consensus_genes', False):
        consensus_file = config['data'].get('consensus_gene_file', 'data/raw/consensus_genes_308.txt')
        logger.info(f"Loading consensus genes from: {consensus_file}")
        with open(consensus_file, 'r') as f:
            consensus_genes = [line.strip() for line in f if line.strip()]
        available_genes = [g for g in consensus_genes if g in expr_raw.index]
        expr_raw = expr_raw.loc[available_genes]
        logger.info(f"After consensus filter: {len(expr_raw)} genes × {expr_raw.shape[1]} samples")
    
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