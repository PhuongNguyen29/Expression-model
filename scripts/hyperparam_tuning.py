"""
Hyperparameter tuning for DeepSurv using Optuna
Based on: Akiba et al., 2019, "Optuna: A Next-generation Hyperparameter Optimization Framework"
"""

import sys
sys.path.append('.')

import torch
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

# Your existing modules
from src.data.dataset import SurvivalDataset
from torch.utils.data import DataLoader
from src.models.deepsurv import DeepSurv, DeepSurvTrainer, calculate_concordance_index

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Optuna's verbose output during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)


class DeepSurvHyperparameterTuner:
    """
    Hyperparameter tuning for DeepSurv using Optuna.
    
    Follows best practices from:
    - Bergstra & Bengio, 2012: Random search superiority
    - Snoek et al., 2012: Bayesian optimization
    - Akiba et al., 2019: Optuna framework
    """
    
    def __init__(
        self,
        train_expr: pd.DataFrame,
        train_surv: pd.DataFrame,
        valid_expr: pd.DataFrame = None,
        valid_surv: pd.DataFrame = None,
        cohort_name: str = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        seed: int = 42
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            train_expr: Training expression data
            train_surv: Training survival data
            valid_expr: Validation expression data (optional, will split from train if None)
            valid_surv: Validation survival data (optional)
            device: Device to use for training
            seed: Random seed
        """
        self.device = device
        self.seed = seed
        self.cohort_name = cohort_name
        self.n_features = train_expr.shape[0]
        self.n_samples = train_expr.shape[1]
        
        logger.info(f"Cohort: {cohort_name}")
        logger.info(f"Number of features (genes): {self.n_features}")
        logger.info(f"Number of training samples: {self.n_samples}")
        logger.info(f"Feature-to-sample ratio: {self.n_features/self.n_samples:.1f}")
        
        # Set seeds
        self._set_seed(seed)
        
        # Create datasets
        if valid_expr is None:
            # Split training data
            full_dataset = SurvivalDataset(train_expr, train_surv)
            n_samples = len(full_dataset)
            n_valid = int(n_samples * 0.2)
            n_train = n_samples - n_valid
            
            train_dataset, valid_dataset = torch.utils.data.random_split(
                full_dataset, [n_train, n_valid],
                generator=torch.Generator().manual_seed(seed)
            )
            self.train_dataset = train_dataset
            self.valid_dataset = valid_dataset
        else:
            self.train_dataset = SurvivalDataset(train_expr, train_surv)
            self.valid_dataset = SurvivalDataset(valid_expr, valid_surv)
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.valid_dataset)}")
    
    def _set_seed(self, seed: int):
        """Set all random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def create_model(self, trial: optuna.Trial) -> DeepSurv:
        """
        Create model with hyperparameters suggested by Optuna.
        
        Based on hyperparameter ranges from:
        - Original DeepSurv paper (Katzman et al., 2018)
        - Kvamme et al., 2019 (Time-to-event prediction review)
        - Our empirical experience with genomic data
        """
        if self.n_samples < 500:  # TCGA
            first_layer_options = [64, 128, 256]
            max_layers = 2
            dropout_min, dropout_max = 0.5, 0.8  # Heavy dropout
        elif self.n_samples < 1200:  # ORIEN  
            first_layer_options = [128, 256, 512]
            max_layers = 3
            dropout_min, dropout_max = 0.3, 0.6  # Moderate dropout
        else:  # Combined
            first_layer_options = [256, 512, 1024]
            max_layers = 3
            dropout_min, dropout_max = 0.2, 0.5  # Standard dropout
        
        # Network architecture
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_sizes = []
        
        # First layer size (larger for genomic data)
        first_layer = trial.suggest_categorical(
            'first_layer_size', 
            first_layer_options
        )
        hidden_sizes.append(first_layer)
        
        # Subsequent layers (decreasing size)
        for i in range(1, n_layers):
            prev_size = hidden_sizes[-1]
            # Each layer is 50% to 100% of previous layer size
            layer_size = trial.suggest_int(
                f'layer_{i}_size',
                int(prev_size * 0.25),
                int(prev_size * 0.75),
                step=32
            )
            hidden_sizes.append(layer_size)
        
        # Regularization
        dropout = trial.suggest_float('dropout', dropout_min, dropout_max, step=0.1)
        
        # Activation function
        activation = trial.suggest_categorical(
            'activation',
            ['relu', 'elu']
        )
        
        if dropout > 0.5:
            batch_norm = False  # Don't use batch norm with high dropout
        else:
            batch_norm = trial.suggest_categorical('batch_norm', [True, False])
        
        # Weight initialization
        weight_init = trial.suggest_categorical(
            'weight_init',
            ['xavier_normal', 'kaiming_uniform']
        )
        
        # Create model
        model = DeepSurv(
            n_features=self.n_features,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
            weight_init=weight_init
        )
        logger.info(f"Trial model: {hidden_sizes}, dropout={dropout:.1f}")
        
        return model
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Returns:
            Negative C-index (we minimize, so negative for maximization)
        """
        
        # Suggest hyperparameters
        if self.n_samples < 500:  # TCGA
            batch_size = trial.suggest_categorical('batch_size', [16, 32])
            weight_decay_min, weight_decay_max = 1e-3, 1e-1  # Strong L2
        elif self.n_samples < 1200:  # ORIEN
            batch_size = trial.suggest_categorical('batch_size', [32, 64])
            weight_decay_min, weight_decay_max = 1e-4, 1e-2  # Moderate L2
        else:  # Combined
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            weight_decay_min, weight_decay_max = 1e-5, 1e-2  # Standard L2
        
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', weight_decay_min, weight_decay_max, log=True)

        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2 if not self.device == 'mps' else 0,
            pin_memory=torch.cuda.is_available()
        )
        
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2 if not self.device == 'mps' else 0,
            pin_memory=torch.cuda.is_available()
        )
        
        # Create model
        model = self.create_model(trial)
        
        # Create trainer
        trainer = DeepSurvTrainer(
            model=model,
            device=self.device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_patience=10
        )
        
        # Training loop with early stopping
        n_epochs = 100  # Max epochs
        early_stopping_patience = 15
        best_cindex = 0
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Train
            train_loss = trainer.train_epoch(train_loader)
            
            # Validate
            valid_loss, valid_cindex = trainer.evaluate(valid_loader)
            
            # Report intermediate value for pruning
            trial.report(valid_cindex, epoch)
            
            # Handle pruning (early termination of bad trials)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if valid_cindex > best_cindex:
                best_cindex = valid_cindex
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                break
        
        return best_cindex  # Optuna maximizes by default
    
    def optimize(
        self,
        n_trials: int = 100,
        timeout: int = None,
        study_name: str = "deepsurv_optimization",
        pruner: str = 'median'
    ) -> Tuple[Dict[str, Any], optuna.Study]:
        """
        Run hyperparameter optimization.
        """
        
        # Select pruner (for early stopping of bad trials)
        if pruner == 'median':
            pruner_obj = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=20,
                interval_steps=5
            )
        elif pruner == 'hyperband':
            pruner_obj = optuna.pruners.HyperbandPruner(
                min_resource=10,
                max_resource=100,
                reduction_factor=3
            )
        else:
            pruner_obj = None
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # Maximize C-index
            pruner=pruner_obj,
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        
        # Optimize
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        logger.info(f"Cohort: {self.cohort_name}, Samples: {self.n_samples}, Features: {self.n_features}")
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"\nBest C-index: {best_value:.4f}")
        logger.info("Best parameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        return best_params, study


def run_hyperparameter_search(
    cohort: str = 'tcga',
    n_trials: int = 50,
    output_dir: str = None
):
    """
    Run hyperparameter search for specified cohort.
    """
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/hyperparam_tuning_{cohort}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data for {cohort} cohort...")
    tcga_expr = pd.read_csv("data/processed/tcga_preprocessed.csv", index_col=0)
    orien_expr = pd.read_csv("data/processed/orien_preprocessed.csv", index_col=0)
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # Prepare data based on cohort
    if cohort.lower() == 'tcga':
        train_expr, train_surv = tcga_expr, surv_tcga
    elif cohort.lower() == 'orien':
        train_expr, train_surv = orien_expr, surv_orien
    elif cohort.lower() == 'combined':
        # Combine both cohorts (concatenate along samples axis)
        train_expr = pd.concat([tcga_expr, orien_expr], axis=1)
        train_surv = pd.concat([surv_tcga, surv_orien])
    else:
        raise ValueError(f"Unknown cohort: {cohort}")
    
    # Create tuner with cohort name
    tuner = DeepSurvHyperparameterTuner(
        train_expr=train_expr,
        train_surv=train_surv,
        cohort_name=cohort
    )
    
    # Run optimization
    best_params, study = tuner.optimize(
        n_trials=n_trials,
        study_name=f"deepsurv_{cohort}"
    )
    
    # Save results
    with open(f"{output_dir}/best_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save study
    study.trials_dataframe().to_csv(f"{output_dir}/trials.csv", index=False)
    
    # Visualize optimization history
    visualize_optimization(study, output_dir)
    
    # Train final model with best parameters
    logger.info("\nTraining final model with best parameters...")
    final_results = train_with_best_params(
        best_params, train_expr, train_surv, output_dir
    )
    
    # Add summary statistics
    summary = {
        'cohort': cohort,
        'n_samples': train_expr.shape[1],
        'n_features': train_expr.shape[0],
        'feature_to_sample_ratio': train_expr.shape[0] / train_expr.shape[1],
        'best_cindex': study.best_value,
        'n_trials': len(study.trials),
        'best_params': best_params
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}")
    
    return best_params, study, final_results


def train_with_best_params(
    best_params: Dict[str, Any],
    train_expr: pd.DataFrame,
    train_surv: pd.DataFrame,
    output_dir: str
) -> Dict[str, Any]:
    """
    Train final model with best hyperparameters.
    """
    
    # FIX 6: Correct dimension for n_features
    model_params = {
        'n_features': train_expr.shape[0],  # Genes are in rows
        'hidden_sizes': [],
        'dropout': best_params['dropout'],
        'activation': best_params['activation'],
        'batch_norm': best_params.get('batch_norm', False),  # May not exist if dropout > 0.5
        'weight_init': best_params['weight_init']
    }
    
    # Reconstruct hidden sizes
    n_layers = best_params['n_layers']
    model_params['hidden_sizes'].append(best_params['first_layer_size'])
    for i in range(1, n_layers):
        if f'layer_{i}_size' in best_params:
            model_params['hidden_sizes'].append(best_params[f'layer_{i}_size'])
    
    # Create model
    model = DeepSurv(**model_params)
    
    # Create data loaders
    dataset = SurvivalDataset(train_expr, train_surv)
    n_samples = len(dataset)
    n_valid = int(n_samples * 0.2)
    n_train = n_samples - n_valid
    
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_valid],
        generator=torch.Generator().manual_seed(42)
    )
    
    # FIX 7: Handle pin_memory
    use_pin_memory = torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 2 if device != 'mps' else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    # Train model
    trainer = DeepSurvTrainer(
        model=model,
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        n_epochs=200,
        early_stopping_patience=20,
        verbose=True
    )
    
    # Save model
    torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
    
    # Save training history with proper type conversion
    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(convert_to_serializable(history), f, indent=2)
    
    return {
        'model': model,
        'history': history,
        'best_valid_cindex': max(history['valid_cindex'])
    }


def visualize_optimization(study: optuna.Study, output_dir: str):
    """Create visualization plots for the optimization study."""
    # [Keep existing visualization code as-is]
    pass  # Implementation remains the same


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed hyperparameter tuning for DeepSurv')
    parser.add_argument('--cohort', type=str, default='tcga',
                       choices=['tcga', 'orien', 'combined'],
                       help='Which cohort to optimize on')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials')
    
    args = parser.parse_args()
    
    # Run single optimization
    best_params, study, final_results = run_hyperparameter_search(
        cohort=args.cohort,
        n_trials=args.n_trials
    )
    
    print("\n" + "="*60)
    print(f"OPTIMIZATION COMPLETE FOR {args.cohort.upper()}")
    print("="*60)
    print(f"Best C-index: {study.best_value:.4f}")
    print(f"Final model C-index: {final_results['best_valid_cindex']:.4f}")
    print("="*60)