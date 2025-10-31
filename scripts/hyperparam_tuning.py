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
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer  # NEW

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
        model_type: str = 'deepsurv',
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

        logger.info(f"Model type: {self.model_type}")
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
                hidden_sizes = [trial.suggest_categorical('single_layer_size', [128, 256])]
            else:
                # Two-layer: only safe combinations
                first = trial.suggest_categorical('first_layer_size', [256, 384])  # Not 512
                second = trial.suggest_categorical('second_layer_size', [64, 128])
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
        if self.model_type == 'deepsurv':
            model = DeepSurv(
                n_features=self.n_features,
                hidden_sizes=hidden_sizes,
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm,
                weight_init=weight_init
            )
        else:  # elastic_deepsurv
            # CRITICAL: Focused search space for elastic net parameters
            # Based on observed issues: alpha too high, need lower learning rate
            
            # Alpha: Regularization strength (REDUCED from default 0.01)
            # Reference: Simon et al., 2011 - typical range 0.0001-0.1
            alpha = trial.suggest_float('alpha', 1e-4, 1e-2, log=True)
            
            # L1 ratio: Balance between L1 (sparsity) and L2 (stability)
            # Reference: Zou & Hastie, 2005
            l1_ratio = trial.suggest_categorical('l1_ratio', [0.5, 0.7, 0.9])
            
            model = ElasticDeepSurv(
                n_features=self.n_features,
                hidden_sizes=hidden_sizes,
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm,
                weight_init=weight_init,
                l1_ratio=l1_ratio,
                alpha=alpha
            )
        
        logger.info(f"Trial model: {hidden_sizes}, dropout={dropout:.2f}")
        if self.model_type == 'elastic_deepsurv':
            logger.info(f"  Elastic net: alpha={trial.params.get('alpha', 'N/A'):.4f}, l1_ratio={trial.params.get('l1_ratio', 'N/A'):.2f}")
        
        return model
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization using stratified k-fold CV.
        
        Returns:
            Mean C-index across all folds
        """
        
        # Training hyperparameters - sample-size-aware
        if self.n_samples < 500:  # TCGA
            batch_size = trial.suggest_categorical('batch_size', [32, 48])
            
            if self.model_type == 'deepsurv':
                weight_decay = trial.suggest_float('weight_decay', 1e-3, 1e-1, log=True)
            else:  # elastic_deepsurv
                weight_decay = 0.0  # CRITICAL: Must be 0 for elastic net
        else:  # ORIEN
            batch_size = trial.suggest_categorical('batch_size', [32, 64])
            
            if self.model_type == 'deepsurv':
                weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
            else:  # elastic_deepsurv
                weight_decay = 0.0  # CRITICAL: Must be 0 for elastic net
        
        # CRITICAL: Learning rate range adjusted based on observed gradient issues
        # Original range: 1e-4 to 1e-2
        # For elastic_deepsurv: favor lower end (1e-5 to 1e-3) due to large gradients
        if self.model_type == 'elastic_deepsurv':
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        else:
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        # Stratified k-fold cross-validation
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
                sampler=train_sampler
            )
            val_loader = DataLoader(
                self.full_dataset,
                batch_size=batch_size,
                sampler=val_sampler
            )
            
            # Create model for this fold
            model = self.create_model(trial)
            
            # Create appropriate trainer
            if self.model_type == 'deepsurv':
                trainer = DeepSurvTrainer(
                    model=model,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    device=self.device
                )
            else:  # elastic_deepsurv
                trainer = ElasticDeepSurvTrainer(
                    model=model,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,  # Will be 0
                    device=self.device
                )
            
            # Train with early stopping
            try:
                history = trainer.fit(
                    train_loader=train_loader,
                    valid_loader=val_loader,
                    n_epochs=100,  # Max epochs
                    early_stopping_patience=15,
                    verbose=False  # Suppress per-epoch output
                )
                
                # Get best validation C-index
                best_cindex = max(history['valid_c_index'])
                cv_scores.append(best_cindex)
                
                logger.info(f"  Fold {fold+1}/{self.n_folds}: C-index = {best_cindex:.4f}")
                
            except Exception as e:
                logger.warning(f"  Fold {fold+1} failed: {e}")
                cv_scores.append(0.5)  # Worst case
                continue
            
            # Report intermediate value for pruning
            trial.report(np.mean(cv_scores), fold)
            
            # Prune if this trial is unpromising
            if trial.should_prune():
                logger.info(f"  Trial pruned at fold {fold+1}")
                raise optuna.TrialPruned()
        
        mean_cindex = np.mean(cv_scores)
        std_cindex = np.std(cv_scores)
        
        logger.info(f"Trial {trial.number}: Mean C-index = {mean_cindex:.4f} ± {std_cindex:.4f}")
        
        return mean_cindex
    
    def optimize(
        self,
        n_trials: int = 50,
        study_name: str = None,
        timeout: int = None
    ) -> Tuple[Dict, optuna.Study]:
        """
        Run Optuna optimization with stratified k-fold CV.
        
        Args:
            n_trials: Number of trials
            study_name: Name for the study
            timeout: Maximum time in seconds (optional)
            
        Returns:
            (best_params, study) tuple
        """
        
        if study_name is None:
            study_name = f"{self.model_type}_{self.cohort_name}"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING HYPERPARAMETER OPTIMIZATION")
        logger.info(f"{'='*60}")
        logger.info(f"Model: {self.model_type}")
        logger.info(f"Cohort: {self.cohort_name}")
        logger.info(f"Strategy: {self.n_folds}-fold stratified CV")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"Device: {self.device}")
        logger.info(f"{'='*60}\n")
        
        # Create study with pruning
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # Maximize C-index
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=2
            )
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Best C-index: {study.best_value:.4f}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best parameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")
        logger.info(f"{'='*60}\n")
        
        return study.best_params, study


def train_final_model_on_full_cohort(
    best_params: Dict[str, Any],
    train_expr: pd.DataFrame,
    train_surv: pd.DataFrame,
    model_type: str,
    cohort_name: str,
    output_dir: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42
) -> Dict[str, Any]:
    """
    Train final model on 100% of cohort data using best hyperparameters.
    
    Args:
        best_params: Best hyperparameters from optimization
        train_expr: Full training expression data
        train_surv: Full training survival data
        model_type: 'deepsurv' or 'elastic_deepsurv'
        cohort_name: Name of cohort
        output_dir: Directory to save model
        device: Device to use
        seed: Random seed
        
    Returns:
        Dictionary with training results
    """
    
    logger.info(f"Training final {model_type} model on full {cohort_name} cohort...")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create dataset and loader
    full_dataset = SurvivalDataset(train_expr, train_surv)
    batch_size = best_params.get('batch_size', 32)
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    # Extract model architecture params
    n_features = train_expr.shape[0]
    
    # Reconstruct architecture
    n_layers = best_params.get('n_layers', 2)
    if n_layers == 1:
        hidden_sizes = [best_params.get('single_layer_size', 256)]
    elif n_layers == 2:
        pattern = best_params.get('pattern_2')
        if pattern:
            hidden_sizes = [int(x) for x in pattern.split('-')]
        else:
            first = best_params.get('first_layer_size', 256)
            second = best_params.get('second_layer_size', 128)
            hidden_sizes = [first, second]
    else:  # n_layers == 3
        pattern = best_params.get('pattern_3')
        if pattern:
            hidden_sizes = [int(x) for x in pattern.split('-')]
        else:
            hidden_sizes = [256, 128, 64]
    
    dropout = best_params.get('dropout', 0.3)
    activation = best_params.get('activation', 'relu')
    batch_norm = best_params.get('batch_norm', True)
    weight_init = best_params.get('weight_init', 'kaiming_uniform')
    
    # Create model
    if model_type == 'deepsurv':
        model = DeepSurv(
            n_features=n_features,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
            weight_init=weight_init
        )
        
        weight_decay = best_params.get('weight_decay', 0.01)
        
    else:  # elastic_deepsurv
        alpha = best_params.get('alpha', 0.001)
        l1_ratio = best_params.get('l1_ratio', 0.7)
        
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
        
        weight_decay = 0.0  # Always 0 for elastic net
    
    # Create trainer
    learning_rate = best_params.get('learning_rate', 0.001)
    
    if model_type == 'deepsurv':
        trainer = DeepSurvTrainer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device
        )
    else:  # elastic_deepsurv
        trainer = ElasticDeepSurvTrainer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device
        )
    
    # Train on full dataset
    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=train_loader,  # Use training data for monitoring
        n_epochs=200,
        early_stopping_patience=30,
        verbose=True
    )
    
    # Save model
    model_path = f"{output_dir}/final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_params': best_params,
        'history': history
    }, model_path)
    
    logger.info(f"Final model saved to {model_path}")
    
    # Get feature importance for elastic_deepsurv
    if model_type == 'elastic_deepsurv':
        gene_names = train_expr.index.tolist()
        importance = model.get_feature_importance(gene_names)
        
        # Save top features
        importance_df = pd.DataFrame(importance, columns=['gene', 'importance'])
        importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
        
        logger.info(f"Feature importance saved to {output_dir}/feature_importance.csv")
        logger.info(f"Top 10 features:")
        for i, (gene, score) in enumerate(importance[:10], 1):
            logger.info(f"  {i}. {gene}: {score:.4f}")
        
        # Get sparsity info
        sparsity_info = model.get_sparsity_info()
        logger.info(f"\nFinal model sparsity: {sparsity_info['sparsity_ratio']:.1%} "
                   f"({sparsity_info['n_zeros']} / {sparsity_info['n_total']} weights)")
    
    results = {
        'final_train_loss': history['train_loss'][-1],
        'best_train_cindex': max(history['train_cindex']),
        'n_epochs_trained': len(history['train_loss']),
        'model_path': model_path
    }
    
    return results


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
    
    # 5. Alpha vs C-index (for elastic_deepsurv)
    if 'params_alpha' in trials.columns:
        ax = axes[1, 1]
        scatter = ax.scatter(trials['params_alpha'], trials['value'], 
                           c=trials['params_l1_ratio'] if 'params_l1_ratio' in trials.columns else 'blue',
                           alpha=0.6, cmap='viridis')
        ax.set_xlabel('Alpha (log scale)')
        ax.set_ylabel('C-index')
        ax.set_xscale('log')
        ax.set_title('Alpha vs Performance')
        ax.grid(True, alpha=0.3)
        if 'params_l1_ratio' in trials.columns:
            plt.colorbar(scatter, ax=ax, label='L1 Ratio')
    elif 'params_dropout' in trials.columns:
        ax = axes[1, 1]
        ax.scatter(trials['params_dropout'], trials['value'], alpha=0.5)
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('C-index')
        ax.set_title('Dropout vs Performance')
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No alpha/dropout data', ha='center', va='center')
    
    # 6. Weight decay or L1 ratio vs C-index
    if 'params_l1_ratio' in trials.columns:
        ax = axes[1, 2]
        ax.scatter(trials['params_l1_ratio'], trials['value'], alpha=0.5)
        ax.set_xlabel('L1 Ratio')
        ax.set_ylabel('C-index')
        ax.set_title('L1 Ratio vs Performance')
        ax.grid(True, alpha=0.3)
    elif 'params_weight_decay' in trials.columns:
        ax = axes[1, 2]
        ax.scatter(trials['params_weight_decay'], trials['value'], alpha=0.5, c=trials['value'], cmap='viridis')
        ax.set_xlabel('Weight Decay (log scale)')
        ax.set_ylabel('C-index')
        ax.set_xscale('log')
        ax.set_title('Weight Decay vs Performance')
        ax.grid(True, alpha=0.3)
        plt.colorbar(ax.collections[0], ax=ax, label='C-index')
    else:
        axes[1, 2].text(0.5, 0.5, 'No regularization data', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimization_plots.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Optimization plots saved to {output_dir}/optimization_plots.png")


def run_hyperparameter_search(
    cohort: str = 'tcga',
    model_type: str = 'deepsurv',  # NEW parameter
    n_trials: int = 50,
    output_dir: str = None
):
    """
    Run hyperparameter search for specified cohort and model type.
    
    Args:
        cohort: Which cohort to optimize ('tcga' or 'orien')
        model_type: Which model ('deepsurv' or 'elastic_deepsurv')
        n_trials: Number of optimization trials
        output_dir: Directory to save results
    """
    
    # Validate inputs
    if cohort.lower() not in ['tcga', 'orien']:
        raise ValueError(f"Cohort must be 'tcga' or 'orien', got: {cohort}")
    
    if model_type.lower() not in ['deepsurv', 'elastic_deepsurv']:
        raise ValueError(f"model_type must be 'deepsurv' or 'elastic_deepsurv', got: {model_type}")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/hyperparam_tuning_{model_type}_{cohort}_{timestamp}"
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
    tuner = SurvivalModelHyperparameterTuner(
        train_expr=train_expr,
        train_surv=train_surv,
        model_type=model_type,
        cohort_name=cohort
    )
    
    # Run optimization with stratified k-fold CV
    best_params, study = tuner.optimize(
        n_trials=n_trials,
        study_name=f"{model_type}_{cohort}"
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
        model_type=model_type,
        cohort_name=cohort,
        output_dir=output_dir
    )
    
    # Save summary
    summary = {
        'cohort': cohort,
        'model_type': model_type,
        'n_samples': train_expr.shape[1],
        'n_features': train_expr.shape[0],
        'feature_to_sample_ratio': train_expr.shape[0] / train_expr.shape[1],
        'cv_best_cindex': study.best_value,
        'n_trials': len(study.trials),
        'best_params': best_params,
        'final_train_loss': final_results['final_train_loss']
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMIZATION COMPLETE FOR {model_type.upper()} ON {cohort.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Best CV C-index: {study.best_value:.4f}")
    logger.info(f"Final training loss: {final_results['final_train_loss']:.4f}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*60}\n")
    
    return best_params, study, final_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for survival models with stratified k-fold CV'
    )
    parser.add_argument(
        '--cohort', 
        type=str, 
        default='tcga',
        choices=['tcga', 'orien'],
        help='Which cohort to optimize (tcga or orien)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='deepsurv',
        choices=['deepsurv', 'elastic_deepsurv'],
        help='Which model to optimize (deepsurv or elastic_deepsurv)'
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
        model_type=args.model_type,
        n_trials=args.n_trials,
        output_dir=args.output_dir
    )