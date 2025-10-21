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
        self.n_features = train_expr.shape[1]
        
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
        
        # Network architecture
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_sizes = []
        
        # First layer size (larger for genomic data)
        first_layer = trial.suggest_categorical(
            'first_layer_size', 
            [128, 256, 512, 1024, 2048]
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
        dropout = trial.suggest_float('dropout', 0.1, 0.7, step=0.1)
        
        # Activation function
        activation = trial.suggest_categorical(
            'activation',
            ['relu', 'elu', 'selu', 'leaky_relu']
        )
        
        # Batch normalization
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])
        
        # Weight initialization
        weight_init = trial.suggest_categorical(
            'weight_init',
            ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']
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
        
        return model
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Returns:
            Negative C-index (we minimize, so negative for maximization)
        """
        
        # Suggest hyperparameters
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
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
        
        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds (optional)
            study_name: Name for the study
            pruner: Pruning strategy ('median', 'hyperband', or None)
        
        Returns:
            best_params: Best hyperparameters found
            study: Optuna study object
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
            sampler=optuna.samplers.TPESampler(seed=self.seed)  # Tree-structured Parzen Estimator
        )
        
        # Optimize
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
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
    cohort: str = 'tcga',  # 'tcga', 'orien', or 'combined'
    n_trials: int = 50,
    output_dir: str = None
):
    """
    Run hyperparameter search for specified cohort.
    
    Args:
        cohort: Which cohort to use for training
        n_trials: Number of optimization trials
        output_dir: Directory to save results
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
        # Combine both cohorts
        train_expr = pd.concat([tcga_expr, orien_expr])
        train_surv = pd.concat([surv_tcga, surv_orien])
    else:
        raise ValueError(f"Unknown cohort: {cohort}")
    
    # Create tuner
    tuner = DeepSurvHyperparameterTuner(
        train_expr=train_expr,
        train_surv=train_surv
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
    
    return best_params, study, final_results


def visualize_optimization(study: optuna.Study, output_dir: str):
    """Create visualization plots for the optimization study."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Optimization history
    ax = axes[0, 0]
    trials = study.trials_dataframe()
    ax.plot(trials.index, trials['value'], 'b-', alpha=0.5, label='All trials')
    best_values = trials['value'].cummax()
    ax.plot(trials.index, best_values, 'r-', linewidth=2, label='Best so far')
    ax.set_xlabel('Trial')
    ax.set_ylabel('C-index')
    ax.set_title('Optimization History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Parameter importance (if enough trials)
    if len(study.trials) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            ax = axes[0, 1]
            params = list(importance.keys())[:10]  # Top 10
            values = list(importance.values())[:10]
            ax.barh(params, values)
            ax.set_xlabel('Importance')
            ax.set_title('Hyperparameter Importance')
        except:
            axes[0, 1].text(0.5, 0.5, 'Insufficient trials for importance analysis',
                           ha='center', va='center')
    
    # 3. Parallel coordinate plot for top trials
    ax = axes[0, 2]
    top_trials = trials.nlargest(10, 'value')
    param_cols = [col for col in top_trials.columns if col.startswith('params_')]
    if param_cols:
        from pandas.plotting import parallel_coordinates
        plot_df = top_trials[param_cols + ['value']].copy()
        plot_df.columns = [col.replace('params_', '') for col in plot_df.columns]
        parallel_coordinates(plot_df, 'value', ax=ax, alpha=0.5)
        ax.set_title('Top 10 Trials - Parameter Relationships')
        ax.legend().remove()
    
    # 4. Learning rate vs C-index
    if 'params_learning_rate' in trials.columns:
        ax = axes[1, 0]
        ax.scatter(trials['params_learning_rate'], trials['value'], alpha=0.5)
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('C-index')
        ax.set_xscale('log')
        ax.set_title('Learning Rate vs Performance')
        ax.grid(True, alpha=0.3)
    
    # 5. Dropout vs C-index
    if 'params_dropout' in trials.columns:
        ax = axes[1, 1]
        ax.scatter(trials['params_dropout'], trials['value'], alpha=0.5)
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('C-index')
        ax.set_title('Dropout vs Performance')
        ax.grid(True, alpha=0.3)
    
    # 6. Architecture size vs C-index
    if 'params_first_layer_size' in trials.columns:
        ax = axes[1, 2]
        ax.scatter(trials['params_first_layer_size'], trials['value'], alpha=0.5)
        ax.set_xlabel('First Layer Size')
        ax.set_ylabel('C-index')
        ax.set_title('Network Size vs Performance')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimization_plots.png", dpi=150)
    plt.close()
    
    logger.info(f"Optimization plots saved to {output_dir}/optimization_plots.png")


def train_with_best_params(
    best_params: Dict[str, Any],
    train_expr: pd.DataFrame,
    train_surv: pd.DataFrame,
    output_dir: str
) -> Dict[str, Any]:
    """
    Train final model with best hyperparameters.
    
    Returns comprehensive results for analysis.
    """
    
    # Extract model architecture params
    model_params = {
        'n_features': train_expr.shape[1],
        'hidden_sizes': [],
        'dropout': best_params['dropout'],
        'activation': best_params['activation'],
        'batch_norm': best_params['batch_norm'],
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=2
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
    
    # Save training history
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return {
        'model': model,
        'history': history,
        'best_valid_cindex': max(history['valid_cindex'])
    }


def compare_configurations():
    """
    Compare different hyperparameter configurations:
    1. Default (from paper)
    2. Optimized for TCGA
    3. Optimized for ORIEN
    4. Optimized for Combined
    """
    
    # Default configuration from DeepSurv paper
    default_config = {
        'hidden_sizes': [512, 256],
        'dropout': 0.4,
        'activation': 'relu',
        'batch_norm': True,
        'weight_init': 'xavier_uniform',
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'batch_size': 32
    }
    
    logger.info("="*60)
    logger.info("HYPERPARAMETER COMPARISON EXPERIMENT")
    logger.info("="*60)
    
    # Run optimization for each cohort
    results = {}
    
    for cohort in ['tcga', 'orien', 'combined']:
        logger.info(f"\nOptimizing for {cohort.upper()} cohort...")
        best_params, study, final_results = run_hyperparameter_search(
            cohort=cohort,
            n_trials=50  # Adjust based on computational budget
        )
        
        results[cohort] = {
            'best_params': best_params,
            'best_cindex': study.best_value,
            'n_trials': len(study.trials),
            'final_model_cindex': final_results['best_valid_cindex']
        }
    
    # Print comparison table
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("="*80)
    print(f"{'Cohort':<15} {'Best C-index':<15} {'Trials':<10} {'Key Differences':<40}")
    print("-"*80)
    
    for cohort, res in results.items():
        key_diffs = []
        if res['best_params']['dropout'] != default_config['dropout']:
            key_diffs.append(f"dropout={res['best_params']['dropout']:.1f}")
        if res['best_params']['learning_rate'] != default_config['learning_rate']:
            key_diffs.append(f"lr={res['best_params']['learning_rate']:.1e}")
        if res['best_params'].get('first_layer_size', 512) != 512:
            key_diffs.append(f"layer1={res['best_params']['first_layer_size']}")
        
        print(f"{cohort.upper():<15} {res['best_cindex']:<15.4f} {res['n_trials']:<10} {', '.join(key_diffs):<40}")
    
    print("="*80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for DeepSurv')
    parser.add_argument('--cohort', type=str, default='tcga',
                       choices=['tcga', 'orien', 'combined'],
                       help='Which cohort to optimize on')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison across all cohorts')
    
    args = parser.parse_args()
    
    if args.compare:
        # Run full comparison experiment
        results = compare_configurations()
    else:
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