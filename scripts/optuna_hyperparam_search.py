"""
Comprehensive Hyperparameter Search with Optuna
================================================

Optimizes ALL hyperparameters including architecture for each cohort:
- Architecture (layers, sizes) - Scaled by cohort size
- Regularization (lambda, l1_ratio) - Adapted to event rate
- Training (dropout, batch_size, learning_rate)

For ORIEN: Uses larger architectures and weaker lambda due to:
  - 3× more samples (1,112 vs 339)
  - Lower event rate (40.5% vs 45.1%)

Author: Phuong Nguyen
Date: 2024-11-08
"""

import sys
import os
from pathlib import Path
import yaml
import logging
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import optuna
from sklearn.model_selection import StratifiedKFold

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_factory import load_dataset_from_config
from src.models.elastic_deepsurv import ElasticDeepSurv
from torch.utils.data import TensorDataset, DataLoader
from src.utils.batch_samplers import StratifiedBatchSampler
from lifelines.utils import concordance_index
from src.utils.proximal_optimizer import create_proximal_optimizer
from src.models.deepsurv import CoxPHLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
    """
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
    
    return strat_bins


class ComprehensiveHyperparameterTuner:
    """
    Comprehensive hyperparameter tuner using Optuna.
    
    Optimizes:
    - Architecture (layers, sizes)
    - Regularization (lambda, l1_ratio)
    - Training (dropout, batch_size, learning_rate)
    """
    
    def __init__(
        self,
        config_path: str,
        cohort: str,
        output_dir: str,
        n_folds: int = 5,
        seed: int = 42
    ):
        """
        Initialize tuner.
        
        Args:
            config_path: Path to config file
            cohort: 'tcga' or 'orien'
            output_dir: Output directory
            n_folds: Number of CV folds
            seed: Random seed
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cohort = cohort.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_folds = n_folds
        self.seed = seed
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self._set_seeds()
        
        logger.info(f"="*60)
        logger.info(f"Comprehensive Hyperparameter Tuner")
        logger.info(f"Cohort: {self.cohort.upper()}")
        logger.info(f"CV Folds: {self.n_folds}")
        logger.info(f"="*60)
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def load_data(self):
        """Load data for hyperparameter search."""
        logger.info("Loading data...")
        
        data = load_dataset_from_config(self.config)
        
        # Select cohort
        if self.cohort == 'tcga':
            expr_data = data['tcga_expr']
            surv_data = data['surv_tcga']
        else:  # orien
            expr_data = data['orien_expr']
            surv_data = data['surv_orien']
        
        # Convert to tensors
        self.X = torch.FloatTensor(expr_data.T.values).to(self.device)
        self.T = torch.FloatTensor(surv_data['time'].values).to(self.device)
        self.E = torch.FloatTensor(surv_data['event'].values).to(self.device)
        
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        
        # Create stratification bins
        self.strat_bins = create_survival_stratification_bins(
            self.T.cpu().numpy(),
            self.E.cpu().numpy(),
            n_time_bins=6  # Increased from 4 to 6 for better fold homogeneity
        )
        
        logger.info(f"{self.cohort.upper()} samples: {self.n_samples}")
        logger.info(f"Features (genes): {self.n_features}")
        logger.info(f"Events: {self.E.sum().item()}/{len(self.E)} ({100*self.E.mean().item():.1f}%)")
        logger.info(f"Stratification bins: {len(np.unique(self.strat_bins))}")
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Suggest hyperparameters based on cohort size.
        
        TCGA (n=339): Smaller networks, standard lambda
        ORIEN (n=1,112): Larger networks, weaker lambda
        """
        params = {}
        
        # Architecture - Scaled by cohort size
        if self.n_samples < 500:  # TCGA
            n_layers = trial.suggest_int('n_layers', 1, 2)
            if n_layers == 1:
                layer1_size = trial.suggest_categorical('layer1_size', [64, 128, 256])
                params['hidden_sizes'] = [layer1_size]
            else:  # 2 layers
                architecture = trial.suggest_categorical(
                    'architecture',
                    ['256-64', '256-128', '128-64', '128-32']
                )
                params['hidden_sizes'] = [int(x) for x in architecture.split('-')]
            
            params['dropout'] = trial.suggest_categorical('dropout', [0.2, 0.3, 0.4])
            params['batch_size'] = trial.suggest_categorical('batch_size', [32, 48])
            
            # Lambda range for TCGA
            params['lambda_val'] = trial.suggest_float('lambda', 0.00005, 0.001, log=True)
            
        else:  # ORIEN
            n_layers = trial.suggest_int('n_layers', 2, 3)
            if n_layers == 2:
                architecture = trial.suggest_categorical(
                    'architecture_2layer',
                    ['256-128', '256-64', '128-64', '128-32']  # First layer ≤ 256 (not 512!)
                )
                params['hidden_sizes'] = [int(x) for x in architecture.split('-')]
            else:  # 3 layers
                architecture = trial.suggest_categorical(
                    'architecture_3layer',
                    ['256-128-64', '256-128-32', '128-64-32']  # First layer ≤ 256 (not 512!)
                )
                params['hidden_sizes'] = [int(x) for x in architecture.split('-')]
            
            params['dropout'] = trial.suggest_categorical('dropout', [0.3, 0.4, 0.5])
            params['batch_size'] = trial.suggest_categorical('batch_size', [48, 64])
            
            # Much weaker lambda range for ORIEN (lower event rate + boundary issue)
            params['lambda_val'] = trial.suggest_float('lambda', 0.000001, 0.00005, log=True)
        
        # Common hyperparameters
        params['l1_ratio'] = trial.suggest_categorical('l1_ratio', [0.3, 0.5, 0.7])  # More Ridge, less Lasso
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
        params['activation'] = 'relu'  # Fixed
        params['batch_norm'] = True    # Fixed
        
        return params
    
    def train_fold(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        params: dict
    ) -> float:
        """
        Train one CV fold with given hyperparameters.
        
        Returns:
            Best validation C-index for this fold
        """
        # Create train/val tensors
        X_train = self.X[train_indices]
        T_train = self.T[train_indices]
        E_train = self.E[train_indices]
        
        X_val = self.X[val_indices]
        T_val = self.T[val_indices]
        E_val = self.E[val_indices]
        
        # Create datasets
        train_dataset = TensorDataset(X_train, T_train, E_train)
        val_dataset = TensorDataset(X_val, T_val, E_val)
        
        # Create dataloaders
        train_sampler = StratifiedBatchSampler(
            events=E_train.cpu().numpy(),
            batch_size=params['batch_size'],
            min_events_per_batch=3,  # Increased from 1 to 3 for more stable gradients
            shuffle=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False
        )
        
        # Create model
        model = ElasticDeepSurv(
            n_features=self.n_features,
            hidden_sizes=params['hidden_sizes'],
            dropout=params['dropout'],
            activation=params['activation'],
            batch_norm=params['batch_norm'],
            alpha=params['lambda_val'],
            l1_ratio=params['l1_ratio']
        ).to(self.device)
        
        # Create optimizer
        optimizer = create_proximal_optimizer(
            model.parameters(),
            optimizer_type='fista',
            lr=params['learning_rate'],
            alpha=0.1,  # Dummy
            l1_ratio=params['l1_ratio'],
            use_group_lasso=True,
            lambda_scale=params['lambda_val']
        )
        
        cox_criterion = CoxPHLoss()
        
        # Training loop
        best_c_index = 0.0
        patience = 15
        patience_counter = 0
        
        for epoch in range(100):
            model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch_data in train_loader:
                X_batch, T_batch, E_batch = batch_data
                
                risk_scores = model(X_batch)
                loss = cox_criterion(risk_scores, T_batch, E_batch)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Validation
            model.eval()
            all_val_risks = []
            all_val_T = []
            all_val_E = []
            
            with torch.no_grad():
                for batch_data in val_loader:
                    X_batch, T_batch, E_batch = batch_data
                    val_risks = model(X_batch)
                    all_val_risks.append(val_risks)
                    all_val_T.append(T_batch)
                    all_val_E.append(E_batch)
            
            all_val_risks = torch.cat(all_val_risks)
            all_val_T = torch.cat(all_val_T)
            all_val_E = torch.cat(all_val_E)
            
            val_c_index = concordance_index(
                all_val_T.cpu().numpy(),
                -all_val_risks.cpu().numpy().flatten(),
                all_val_E.cpu().numpy()
            )
            
            # Track best
            if val_c_index > best_c_index:
                best_c_index = val_c_index
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_c_index
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Returns:
            Mean CV C-index across all folds
        """
        # Suggest hyperparameters
        params = self.suggest_hyperparameters(trial)
        
        # Log trial info
        logger.info(f"\nTrial {trial.number}:")
        logger.info(f"  Architecture: {params['hidden_sizes']}")
        logger.info(f"  Lambda: {params['lambda_val']:.6f}")
        logger.info(f"  L1_ratio: {params['l1_ratio']}")
        logger.info(f"  Dropout: {params['dropout']}")
        logger.info(f"  Batch size: {params['batch_size']}")
        logger.info(f"  Learning rate: {params['learning_rate']:.6f}")
        
        # Cross-validation
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.seed
        )
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X.cpu().numpy(), self.strat_bins)):
            
            try:
                fold_c_index = self.train_fold(train_idx, val_idx, params)
                cv_scores.append(fold_c_index)
                logger.info(f"  Fold {fold+1}/{self.n_folds}: C-index = {fold_c_index:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed: {e}")
                cv_scores.append(0.5)
            
            # Report intermediate value for pruning
            trial.report(np.mean(cv_scores), fold)
            if trial.should_prune():
                logger.info(f"  Trial pruned at fold {fold+1}")
                raise optuna.TrialPruned()
        
        mean_c_index = np.mean(cv_scores)
        std_c_index = np.std(cv_scores)
        
        logger.info(f"  Mean CV C-index: {mean_c_index:.4f} ± {std_c_index:.4f}")
        
        return mean_c_index
    
    def optimize(self, n_trials: int = 50):
        """
        Run Optuna optimization.
        
        Args:
            n_trials: Number of trials
            
        Returns:
            best_params, study
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING OPTUNA OPTIMIZATION")
        logger.info(f"{'='*60}")
        logger.info(f"Cohort: {self.cohort.upper()}")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"CV Folds: {self.n_folds}")
        logger.info(f"{'='*60}\n")
        
        study = optuna.create_study(
            study_name=f"elastic_deepsurv_{self.cohort}",
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=2
            )
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
        logger.info(f"\nBest parameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Save results
        self._save_results(study)
        
        return study.best_params, study
    
    def _save_results(self, study: optuna.Study):
        """Save optimization results."""
        # Best parameters
        with open(self.output_dir / 'best_params.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # All trials
        trials_df = study.trials_dataframe()
        trials_df.to_csv(self.output_dir / 'trials.csv', index=False)
        
        # Summary
        summary = {
            'cohort': self.cohort,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_folds': self.n_folds,
            'best_cv_cindex': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'n_completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nResults saved to: {self.output_dir}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive hyperparameter search with Optuna'
    )
    parser.add_argument(
        '--cohort',
        type=str,
        required=True,
        choices=['tcga', 'orien'],
        help='Cohort to optimize: tcga or orien'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/experiments/alpha_investigation.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of Optuna trials'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: results/optuna_{cohort}_{timestamp})'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of CV folds'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/optuna_{args.cohort}_{timestamp}"
    
    # Initialize tuner
    tuner = ComprehensiveHyperparameterTuner(
        config_path=args.config,
        cohort=args.cohort,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        seed=args.seed
    )
    
    # Load data
    tuner.load_data()
    
    # Run optimization
    best_params, study = tuner.optimize(n_trials=args.n_trials)
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ Hyperparameter search complete!")
    logger.info(f"Results: {args.output_dir}")
    logger.info(f"Best C-index: {study.best_value:.4f}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
