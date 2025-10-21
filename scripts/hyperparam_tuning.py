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
from sklearn.model_selection import StratifiedKFold

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
        # valid_expr: pd.DataFrame = None,
        # valid_surv: pd.DataFrame = None,
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
        
        self.full_dataset = SurvivalDataset(train_expr, train_surv)
        self.events = train_surv['event'].values  # For stratification
        self.use_kfold = True  # Flag to use k-fold
        self.n_folds = 5

        
        logger.info(f"Cohort: {cohort_name}")
        logger.info(f"Number of features (genes): {self.n_features}")
        logger.info(f"Number of training samples: {self.n_samples}")
        logger.info(f"Feature-to-sample ratio: {self.n_features/self.n_samples:.1f}")
        
        # Set seeds
        self._set_seed(seed)
        
        # # Create datasets
        # if valid_expr is None:
        #     # Split training data
        #     full_dataset = SurvivalDataset(train_expr, train_surv)
        #     n_samples = len(full_dataset)
        #     n_valid = int(n_samples * 0.2)
        #     n_train = n_samples - n_valid
            
        #     train_dataset, valid_dataset = torch.utils.data.random_split(
        #         full_dataset, [n_train, n_valid],
        #         generator=torch.Generator().manual_seed(seed)
        #     )
        #     self.train_dataset = train_dataset
        #     self.valid_dataset = valid_dataset
        # else:
        #     self.train_dataset = SurvivalDataset(train_expr, train_surv)
        #     self.valid_dataset = SurvivalDataset(valid_expr, valid_surv)
        
        # logger.info(f"Training samples: {len(self.train_dataset)}")
        # logger.info(f"Validation samples: {len(self.valid_dataset)}")
    
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
        """
        """Create model with sample-size-aware constraints."""
    
        if self.n_samples < 500:  # TCGA - keep it simple
            n_layers = trial.suggest_int('n_layers', 1, 2)
            
            if n_layers == 1:
                hidden_sizes = [trial.suggest_categorical('layer_0', [128, 256])]
            else:
                # Two-layer: only safe combinations
                first = trial.suggest_categorical('layer_0', [256, 384])  # Not 512
                second = trial.suggest_categorical('layer_1', [64, 128])
                hidden_sizes = [first, second]
            
            dropout = trial.suggest_categorical('dropout', [0.2, 0.3, 0.4])
            # weight_decay_range = (1e-3, 1e-1)
            
        else :  # ORIEN
            n_layers = trial.suggest_int('n_layers', 2, 3)
            if n_layers == 2:
                pattern = trial.suggest_categorical('pattern_2', ['512-128', '384-96', '256-64'])
                hidden_sizes = [int(x) for x in pattern.split('-')]
            else:  # n_layers == 3
                pattern = trial.suggest_categorical('pattern_3', ['512-256-64', '384-192-48', '256-128-32'])
                hidden_sizes = [int(x) for x in pattern.split('-')]
            
            dropout = trial.suggest_categorical('dropout', [0.3, 0.4, 0.5])
        # weight_decay_range = (1e-4, 1e-2)
  
        # # Network architecture
        # n_layers = trial.suggest_int('n_layers', 1, 4)
        # hidden_sizes = []
        
        # # First layer size (larger for genomic data)
        # first_layer = trial.suggest_categorical(
        #     'first_layer_size', 
        #     first_layer_options
        # )
        # hidden_sizes.append(first_layer)
        
        # # Subsequent layers (decreasing size)
        # for i in range(1, n_layers):
        #     prev_size = hidden_sizes[-1]
        #     # Each layer is 50% to 100% of previous layer size
        #     layer_size = trial.suggest_int(
        #         f'layer_{i}_size',
        #         int(prev_size * 0.25),
        #         int(prev_size * 0.75),
        #         step=32
        #     )
        #     hidden_sizes.append(layer_size)
        
        activation = trial.suggest_categorical('activation', ['relu', 'elu'])
        batch_norm = trial.suggest_categorical('batch_norm', [True, False]) if dropout <= 0.4 else False
        weight_init = trial.suggest_categorical('weight_init', ['xavier_normal', 'kaiming_uniform'])
    
            
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
        Objective function for Optuna optimization using stratified k-fold CV.
        
        Returns:
            Mean C-index across all folds
        """
        
        # Suggest training hyperparameters ONCE per trial
        if self.n_samples < 500:  # TCGA
            batch_size = trial.suggest_categorical('batch_size', [32, 48])
            weight_decay = trial.suggest_float('weight_decay', 1e-3, 1e-1, log=True)  # Strong L2
        else:  # ORIEN
            batch_size = trial.suggest_categorical('batch_size', [32, 64])
            weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)  # Moderate L2
        
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        cv_scores = []
            
        for fold, (train_idx, val_idx) in enumerate(skf.split(
            range(len(self.full_dataset)), self.events
        )):
            # Reset seed for reproducibility
            self._set_seed(self.seed + fold)
            
            # Create data loaders for this fold
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                self.full_dataset, 
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=0,
                drop_last=False
            )
            
            val_loader = DataLoader(
                self.full_dataset,
                batch_size=batch_size, 
                sampler=val_sampler,
                num_workers=0,
                drop_last=False
            )
            
            # Create fresh model for this fold
            model = self.create_model(trial)
            
            # Create trainer with complete parameters
            trainer = DeepSurvTrainer(
                model=model,
                device=self.device,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                scheduler_patience=5  # Shorter patience for CV
            )
            
            # Train this fold with early stopping
            best_fold_cindex = 0
            patience_counter = 0
            max_patience = 10
            
            for epoch in range(50):  # Max 50 epochs per fold for efficiency
                train_loss = trainer.train_epoch(train_loader)
                _, val_cindex = trainer.evaluate(val_loader)
                
                # Track best C-index for this fold
                if val_cindex > best_fold_cindex:
                    best_fold_cindex = val_cindex
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping for this fold
                if patience_counter >= max_patience:
                    break
                
                # Optuna pruning (optional - prunes bad trials early)
                trial.report(val_cindex, fold * 50 + epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            cv_scores.append(best_fold_cindex)
            logger.info(f"  Fold {fold+1}/{self.n_folds}: C-index = {best_fold_cindex:.4f}")
        
        # Return mean C-index across folds (standard practice)
        mean_cindex = np.mean(cv_scores)
        std_cindex = np.std(cv_scores)
        logger.info(f"Trial complete: Mean C-index = {mean_cindex:.4f} ± {std_cindex:.4f}")
        
        return mean_cindex
    
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


def train_final_model_on_full_cohort(
    best_params: Dict[str, Any],
    train_expr: pd.DataFrame,
    train_surv: pd.DataFrame,
    cohort_name: str,
    output_dir: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Any]:
    """
    Train final model on 100% of source cohort for bidirectional validation.
    
    This is the correct protocol for transfer learning:
    1. Use k-fold CV to select hyperparameters (done in optimize())
    2. Train final model on ALL source data (this function)
    3. Test on 100% of target cohort (done separately in evaluation script)
    
    Args:
        best_params: Best hyperparameters from optimization
        train_expr: Full training expression data
        train_surv: Full training survival data
        cohort_name: Name of cohort
        output_dir: Directory to save model
        device: Device to train on
        
    Returns:
        Dictionary with model and training history
    """
    
    logger.info(f"\nTraining final model on 100% of {cohort_name} cohort...")
    logger.info(f"Samples: {train_expr.shape[1]}, Features: {train_expr.shape[0]}")
    
    # Reconstruct model architecture from best_params
    hidden_sizes = []
    n_layers = best_params['n_layers']
    
    # Handle different parameter naming based on architecture
    if 'layer_0' in best_params:
        # TCGA style: individual layer parameters
        for i in range(n_layers):
            layer_key = f'layer_{i}'
            if layer_key in best_params:
                hidden_sizes.append(best_params[layer_key])
    elif 'pattern_2' in best_params:
        # ORIEN style: pattern for 2 layers
        hidden_sizes = [int(x) for x in best_params['pattern_2'].split('-')]
    elif 'pattern_3' in best_params:
        # ORIEN style: pattern for 3 layers
        hidden_sizes = [int(x) for x in best_params['pattern_3'].split('-')]
    else:
        raise ValueError(f"Cannot reconstruct architecture from best_params: {best_params}")
    
    # Create model with best hyperparameters
    model = DeepSurv(
        n_features=train_expr.shape[0],
        hidden_sizes=hidden_sizes,
        dropout=best_params['dropout'],
        activation=best_params['activation'],
        batch_norm=best_params.get('batch_norm', False),
        weight_init=best_params['weight_init']
    )
    
    logger.info(f"Model architecture: {hidden_sizes}")
    logger.info(f"Dropout: {best_params['dropout']}, Activation: {best_params['activation']}")
    
    # Create dataset and loader (use ALL data, no validation split)
    full_dataset = SurvivalDataset(train_expr, train_surv)
    train_loader = DataLoader(
        full_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    # Create trainer
    trainer = DeepSurvTrainer(
        model=model,
        device=device,
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        scheduler_patience=10
    )
    
    # Train on full data (monitor training loss only, no validation)
    n_epochs = 200
    train_losses = []
    
    logger.info("Training on full cohort (no validation split)...")
    for epoch in range(n_epochs):
        train_loss = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs}: Training Loss = {train_loss:.4f}")
    
    # Save model
    model_path = f"{output_dir}/final_model_{cohort_name}.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'architecture': hidden_sizes,
        'hyperparameters': best_params
    }
    
    with open(f"{output_dir}/training_history_{cohort_name}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return {
        'model': model,
        'history': history,
        'final_train_loss': train_losses[-1]
    }


# def run_hyperparameter_search(
#     cohort: str = 'tcga',
#     n_trials: int = 50,
#     output_dir: str = None
# ):
#     """
#     Run hyperparameter search for specified cohort.
#     """
    
#     # Create output directory
#     if output_dir is None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_dir = f"results/hyperparam_tuning_{cohort}_{timestamp}"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load data
#     logger.info(f"Loading data for {cohort} cohort...")
#     tcga_expr = pd.read_csv("data/processed/tcga_preprocessed.csv", index_col=0)
#     orien_expr = pd.read_csv("data/processed/orien_preprocessed.csv", index_col=0)
#     surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
#     surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
#     # Prepare data based on cohort
#     if cohort.lower() == 'tcga':
#         train_expr, train_surv = tcga_expr, surv_tcga
#     elif cohort.lower() == 'orien':
#         train_expr, train_surv = orien_expr, surv_orien
#     elif cohort.lower() == 'combined':
#         # Combine both cohorts (concatenate along samples axis)
#         train_expr = pd.concat([tcga_expr, orien_expr], axis=1)
#         train_surv = pd.concat([surv_tcga, surv_orien])
#     else:
#         raise ValueError(f"Unknown cohort: {cohort}")
    
#     # Create tuner with cohort name
#     tuner = DeepSurvHyperparameterTuner(
#         train_expr=train_expr,
#         train_surv=train_surv,
#         cohort_name=cohort
#     )
    
#     # Run optimization
#     best_params, study = tuner.optimize(
#         n_trials=n_trials,
#         study_name=f"deepsurv_{cohort}"
#     )
    
#     # Save results
#     with open(f"{output_dir}/best_params.json", 'w') as f:
#         json.dump(best_params, f, indent=2)
    
#     # Save study
#     study.trials_dataframe().to_csv(f"{output_dir}/trials.csv", index=False)
    
#     # Visualize optimization history
#     visualize_optimization(study, output_dir)
    
#     # Train final model with best parameters
#     logger.info("\nTraining final model with best parameters...")
#     final_results = train_with_best_params(
#         best_params, train_expr, train_surv, output_dir
#     )
    
#     # Add summary statistics
#     summary = {
#         'cohort': cohort,
#         'n_samples': train_expr.shape[1],
#         'n_features': train_expr.shape[0],
#         'feature_to_sample_ratio': train_expr.shape[0] / train_expr.shape[1],
#         'best_cindex': study.best_value,
#         'n_trials': len(study.trials),
#         'best_params': best_params
#     }
    
#     with open(f"{output_dir}/summary.json", 'w') as f:
#         json.dump(summary, f, indent=2)
    
#     logger.info(f"\nResults saved to {output_dir}")
    
#     return best_params, study, final_results

def visualize_optimization(study: optuna.Study, output_dir: str):
    """
    Create visualization plots for the optimization study.
    
    Based on: Akiba et al., 2019 - Optuna visualization best practices
    """
    
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
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'Insufficient trials\nfor importance analysis\n({len(study.trials)} trials)',
                           ha='center', va='center')
    
    # 3. Distribution of trial values
    ax = axes[0, 2]
    ax.hist(trials['value'], bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(study.best_value, color='r', linestyle='--', linewidth=2, label='Best')
    ax.set_xlabel('C-index')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Trial Performances')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Learning rate vs C-index
    if 'params_learning_rate' in trials.columns:
        ax = axes[1, 0]
        ax.scatter(trials['params_learning_rate'], trials['value'], alpha=0.5)
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('C-index')
        ax.set_xscale('log')
        ax.set_title('Learning Rate vs Performance')
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No learning_rate data', ha='center', va='center')
    
    # 5. Dropout vs C-index
    if 'params_dropout' in trials.columns:
        ax = axes[1, 1]
        ax.scatter(trials['params_dropout'], trials['value'], alpha=0.5)
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('C-index')
        ax.set_title('Dropout vs Performance')
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No dropout data', ha='center', va='center')
    
    # 6. Weight decay vs C-index
    if 'params_weight_decay' in trials.columns:
        ax = axes[1, 2]
        ax.scatter(trials['params_weight_decay'], trials['value'], alpha=0.5, c=trials['value'], cmap='viridis')
        ax.set_xlabel('Weight Decay (log scale)')
        ax.set_ylabel('C-index')
        ax.set_xscale('log')
        ax.set_title('Weight Decay vs Performance')
        ax.grid(True, alpha=0.3)
        plt.colorbar(ax.collections[0], ax=ax, label='C-index')
    else:
        axes[1, 2].text(0.5, 0.5, 'No weight_decay data', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimization_plots.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Optimization plots saved to {output_dir}/optimization_plots.png")


def run_hyperparameter_search(
    cohort: str = 'tcga',
    n_trials: int = 50,
    output_dir: str = None
):
    """
    Run hyperparameter search for specified cohort.
    
    Args:
        cohort: Which cohort to optimize ('tcga' or 'orien')
        n_trials: Number of optimization trials
        output_dir: Directory to save results
    """
    
    # Validate cohort
    if cohort.lower() not in ['tcga', 'orien']:
        raise ValueError(f"Cohort must be 'tcga' or 'orien', got: {cohort}")
    
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
    
    # Select data based on cohort
    if cohort.lower() == 'tcga':
        train_expr, train_surv = tcga_expr, surv_tcga
    else:  # orien
        train_expr, train_surv = orien_expr, surv_orien
    
    # Create tuner
    tuner = DeepSurvHyperparameterTuner(
        train_expr=train_expr,
        train_surv=train_surv,
        cohort_name=cohort
    )
    
    # Run optimization with stratified k-fold CV
    best_params, study = tuner.optimize(
        n_trials=n_trials,
        study_name=f"deepsurv_{cohort}"
    )
    
    # Save optimization results
    with open(f"{output_dir}/best_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    study.trials_dataframe().to_csv(f"{output_dir}/trials.csv", index=False)
    
    # Visualize optimization
    visualize_optimization(study, output_dir)
    
    # Train final model on 100% of cohort
    logger.info("\n" + "="*60)
    logger.info("TRAINING FINAL MODEL ON FULL COHORT")
    logger.info("="*60)
    
    final_results = train_final_model_on_full_cohort(
        best_params=best_params,
        train_expr=train_expr,
        train_surv=train_surv,
        cohort_name=cohort,
        output_dir=output_dir
    )
    
    # Save summary
    summary = {
        'cohort': cohort,
        'n_samples': train_expr.shape[1],
        'n_features': train_expr.shape[0],
        'feature_to_sample_ratio': train_expr.shape[0] / train_expr.shape[1],
        'cv_best_cindex': study.best_value,  # From k-fold CV
        'n_trials': len(study.trials),
        'best_params': best_params,
        'final_train_loss': final_results['final_train_loss']
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMIZATION COMPLETE FOR {cohort.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Best CV C-index: {study.best_value:.4f}")
    logger.info(f"Final training loss: {final_results['final_train_loss']:.4f}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*60}\n")
    
    return best_params, study, final_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for DeepSurv with stratified k-fold CV'
    )
    parser.add_argument(
        '--cohort', 
        type=str, 
        default='tcga',
        choices=['tcga', 'orien'],
        help='Which cohort to optimize (tcga or orien)'
    )
    parser.add_argument(
        '--n_trials', 
        type=int, 
        default=50,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Run optimization
    best_params, study, final_results = run_hyperparameter_search(
        cohort=args.cohort,
        n_trials=args.n_trials,
        output_dir=args.output_dir
    )