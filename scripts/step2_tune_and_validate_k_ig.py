"""
Step 2 (IG): K-Selection with Per-K Hyperparameter Tuning using Integrated Gradients

This script uses Integrated Gradients (IG) importance scores instead of L2 norm
for gene ranking and consensus selection.

Key differences from L2-based step2_tune_and_validate_k.py:
1. Loads IG aggregated rankings from results_v2/06_importance_methods/aggregated/
2. Uses 'ig_mean' column for gene importance ranking
3. Outputs to results_v2/02b_biomarker_discovery_ig/
4. Uses EPV-aware architecture constraints (see EPV section below)
5. Implements Option 2: CV-derived epochs for fair comparison with Cox

=============================================================================
OPTION 2: CV-DERIVED EPOCHS FOR FAIR COX COMPARISON
=============================================================================

Problem: How to fairly compare DeepSurv with Cox elastic net?
- Cox: Uses 100% training data with built-in CV for lambda selection
- DeepSurv (naive): Uses 80% for training, 20% for early stopping → unfair

Solution (Option 2):
1. During CV hyperparameter tuning, capture best_epoch from each fold
2. Compute mean best_epoch across folds for the best trial
3. Use this CV-derived epoch count when training on 100% of source data
4. This matches Cox's use of full data with principled stopping

Benefits:
- Uses 100% of source data (same as Cox)
- Stopping criterion derived from same CV procedure used for hyperparameters
- Cross-cohort test is true external validation (no information leakage)
- Fair comparison: both methods use full training data

Reference: Katzman et al. (2018) DeepSurv - methodology for fair comparison
=============================================================================

For each k-value:
    1. Extract top-k genes from aggregated IG rankings (TCGA and ORIEN)
    2. Find consensus genes (intersection)
    3. Hyperparameter tuning on TCGA with proper CV preprocessing
       → Captures CV-derived best_epoch for each cohort
    4. Hyperparameter tuning on ORIEN with proper CV preprocessing
       → Captures CV-derived best_epoch for each cohort
    5. Cross-cohort validation with CV-derived epochs (Option 2)

Based on:
- Sundararajan et al. (2017, ICML) - Integrated Gradients
- Bergstra & Bengio (2012, JMLR) - Hyperparameter optimization
- Bernau et al. (2014, Bioinformatics) - Cross-study validation
- Harrell (2001) - EPV guidelines for survival models
- van der Ploeg et al. (2014, BMC Med Res Methodol) - EPV in prediction models
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from typing import List, Dict, Tuple

import torch
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import optuna
from sklearn.model_selection import StratifiedKFold
import json
import yaml
import matplotlib.pyplot as plt

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
    
    logger.info(f"Created {len(np.unique(strat_bins))} stratification bins")
    return strat_bins


class LeakageFreeKSelectionTuner:
    """
    Hyperparameter tuner for k-selection with proper CV preprocessing.
    Based on LeakageFreeHyperparameterTuner from hyperparam_tuning_elastic_FIXED.py
    """
    
    def __init__(
        self,
        train_expr_raw: pd.DataFrame,
        train_surv: pd.DataFrame,
        config: dict,
        cohort_name: str = None,
        k_value: int = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        seed: int = 42
    ):
        self.train_expr_raw = train_expr_raw
        self.train_surv = train_surv
        self.config = config
        self.cohort_name = cohort_name
        self.k_value = k_value
        self.device = device
        self.seed = seed
        
        self.n_samples = train_expr_raw.shape[1]
        self.n_genes_raw = train_expr_raw.shape[0]
        
        # For stratification
        self.events = train_surv['event'].values
        self.times = train_surv['time'].values
        
        # Create stratification bins
        self.strat_bins = create_survival_stratification_bins(
            self.times,
            self.events,
            n_time_bins=4
        )
        
        self.n_folds = 5
        
        logger.info(f"LeakageFreeKSelectionTuner initialized")
        logger.info(f"  Cohort: {cohort_name}, k={k_value}")
        logger.info(f"  Samples: {self.n_samples}")
        logger.info(f"  Genes: {self.n_genes_raw}")
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
    
    def preprocess_fold(self, train_indices: np.ndarray, val_indices: np.ndarray):
        """Preprocess one CV fold with proper train/test separation."""
        # Get sample names
        train_samples = self.train_expr_raw.columns[train_indices]
        val_samples = self.train_expr_raw.columns[val_indices]
        
        # Extract raw data
        train_fold_raw = self.train_expr_raw[train_samples]
        val_fold_raw = self.train_expr_raw[val_samples]
        
        # Create preprocessor
        fold_preprocessor = GeneExpressionPreprocessor(self.config)
        
        # CRITICAL: Fit preprocessor ONLY on training fold
        train_processed = fold_preprocessor.fit_transform_single_cohort(
            train_fold_raw,
            cohort_name=f"{self.cohort_name}_train"
        )
        
        # Transform validation fold using training parameters
        val_processed = fold_preprocessor.transform_single_cohort(val_fold_raw)
        
        # Get survival data
        train_surv_fold = self.train_surv.loc[train_samples]
        val_surv_fold = self.train_surv.loc[val_samples]
        
        # Create datasets
        train_dataset = SurvivalDataset(train_processed, train_surv_fold)
        val_dataset = SurvivalDataset(val_processed, val_surv_fold)
        
        n_features = train_processed.shape[0]
        train_events = train_surv_fold['event'].values
        
        return train_dataset, val_dataset, n_features, train_events
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective with proper CV preprocessing."""
        # =================================================================
        # EPV-AWARE ARCHITECTURE SELECTION
        # =================================================================
        # Based on Harrell (2001) "Regression Modeling Strategies"
        # and van der Ploeg et al. (2014) BMC Med Res Methodol
        #
        # Key constraints:
        # - TCGA: ~153 events, need EPV >= 10 ideally
        # - ORIEN: ~450 events, more flexibility
        # - Consensus genes (m): 20-100 features
        # - Architecture should scale with m, not exceed event count
        # =================================================================
        
        m = self.n_genes_raw  # Number of consensus genes (input features)
        n_events = int(self.events.sum())
        
        # Calculate maximum reasonable parameters (target EPV >= 5)
        # Being conservative since elastic net provides some regularization
        max_params_epv5 = n_events // 5
        
        logger.debug(f"  m={m} genes, {n_events} events, max_params (EPV=5): {max_params_epv5}")
        
        if self.n_samples < 500:  # TCGA (~153 events)
            # Very conservative: only 1-layer architectures
            # Max params ~ m * h1, so h1 <= max_params / m
            max_h1 = min(64, max(8, max_params_epv5 // m))
            
            # Define options based on m
            if m <= 40:
                layer_options = [8, 16, 24]
            elif m <= 60:
                layer_options = [16, 24, 32]
            elif m <= 80:
                layer_options = [24, 32, 48]
            else:  # m > 80
                layer_options = [32, 48, 64]
            
            # Filter by max_h1
            layer_options = [h for h in layer_options if h <= max_h1]
            if not layer_options:
                layer_options = [8]  # Fallback minimum
            
            layer1_size = trial.suggest_categorical('layer1_size', layer_options)
            hidden_sizes = [layer1_size]
            
            # Stronger regularization for small sample
            alpha = trial.suggest_float('alpha', 1e-4, 5e-3, log=True)
            
        else:  # ORIEN (~450 events)
            # More flexibility but still conservative
            if m <= 40:
                # Small feature set: 1 layer only
                layer_options = [16, 24, 32, 48]
                layer1_size = trial.suggest_categorical('layer1_size', layer_options)
                hidden_sizes = [layer1_size]
            elif m <= 70:
                # Medium feature set: 1 or 2 layers
                n_layers = trial.suggest_int('n_layers', 1, 2)
                if n_layers == 1:
                    layer_options = [32, 48, 64]
                    layer1_size = trial.suggest_categorical('layer1_size', layer_options)
                    hidden_sizes = [layer1_size]
                else:
                    architecture = trial.suggest_categorical(
                        'architecture_2layer',
                        ['32-16', '48-24', '64-32']
                    )
                    hidden_sizes = [int(x) for x in architecture.split('-')]
            else:  # m > 70
                # Larger feature set: 1 or 2 layers
                n_layers = trial.suggest_int('n_layers', 1, 2)
                if n_layers == 1:
                    layer_options = [48, 64, 96]
                    layer1_size = trial.suggest_categorical('layer1_size', layer_options)
                    hidden_sizes = [layer1_size]
                else:
                    architecture = trial.suggest_categorical(
                        'architecture_2layer',
                        ['48-24', '64-32', '96-48']
                    )
                    hidden_sizes = [int(x) for x in architecture.split('-')]
            
            # Moderate regularization
            alpha = trial.suggest_float('alpha', 5e-5, 1e-3, log=True)
        
        # Log architecture choice
        total_params = m * hidden_sizes[0]
        if len(hidden_sizes) > 1:
            total_params += hidden_sizes[0] * hidden_sizes[1]
        epv_estimate = n_events / total_params
        logger.debug(f"  Architecture: {hidden_sizes}, ~{total_params} params, EPV~{epv_estimate:.2f}")
        
        # Common hyperparameters
        dropout = trial.suggest_float('dropout', 0.3, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.3, 0.9)
        batch_size = trial.suggest_categorical('batch_size', [32, 64])
        activation = trial.suggest_categorical('activation', ['relu', 'elu'])
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])
        
        # Force Kaiming if BatchNorm is enabled
        if batch_norm:
            weight_init = 'kaiming_normal'
        else:
            weight_init = trial.suggest_categorical('weight_init', ['xavier_normal', 'kaiming_normal'])
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        cv_scores = []
        cv_best_epochs = []  # Track best epoch from each fold for Option 2
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(
            np.arange(self.n_samples),
            self.strat_bins
        )):
            logger.debug(f"  Fold {fold+1}/{self.n_folds}...")
            
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
                    n_epochs=100,
                    early_stopping_patience=20,
                    verbose=False
                )
                
                best_cindex = max(history['valid_c_index'])
                cv_scores.append(best_cindex)
                
                # Capture best_epoch for CV-derived stopping (Option 2)
                # best_epoch is 1-indexed from trainer.fit()
                fold_best_epoch = history.get('best_epoch', len(history['train_loss']))
                cv_best_epochs.append(fold_best_epoch)
                
            except Exception as e:
                logger.warning(f"    Fold {fold+1} failed: {e}")
                cv_scores.append(0.5)
                cv_best_epochs.append(50)  # Default fallback
                continue
            
            # Report for pruning
            trial.report(np.mean(cv_scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        mean_cindex = np.mean(cv_scores)
        std_cindex = np.std(cv_scores)
        
        # Store CV best epochs as trial attribute (for Option 2: CV-derived epochs)
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
            
            logger.info(f"Trial {trial.number}: C-index={mean_cindex:.4f} ± {std_cindex:.4f}, "
                       f"Sparsity={sparsity_ratio:.1%}, Objective={objective_value:.4f}")
            
            return objective_value
        except Exception as e:
            logger.warning(f"Could not compute sparsity: {e}")
        
        logger.info(f"Trial {trial.number}: {mean_cindex:.4f} ± {std_cindex:.4f}")
        return mean_cindex
    
    def optimize(self, n_trials: int = 50):
        """
        Run optimization.
        
        Returns:
            best_params: Best hyperparameters
            study: Optuna study object
            cv_epochs_info: Dict with CV-derived epoch information for Option 2
        """
        study_name = f"k{self.k_value}_{self.cohort_name}_IG"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"HYPERPARAMETER OPTIMIZATION (k={self.k_value}, {self.cohort_name}, IG-based)")
        logger.info(f"{'='*60}")
        logger.info(f"Samples: {self.n_samples}, Genes: {self.n_genes_raw}")
        logger.info(f"CV: {self.n_folds}-fold stratified, Trials: {n_trials}")
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
        
        # Extract CV-derived epochs from best trial (Option 2)
        best_trial = study.best_trial
        cv_best_epochs = best_trial.user_attrs.get('cv_best_epochs', [50] * self.n_folds)
        mean_best_epoch = best_trial.user_attrs.get('mean_best_epoch', 50)
        
        cv_epochs_info = {
            'cv_best_epochs': cv_best_epochs,
            'mean_best_epoch': mean_best_epoch,
            'std_best_epoch': float(np.std(cv_best_epochs)),
            'min_best_epoch': min(cv_best_epochs),
            'max_best_epoch': max(cv_best_epochs)
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"Best CV C-index: {study.best_value:.4f}")
        logger.info(f"Best parameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"\nCV-derived epochs (Option 2 for fair comparison with Cox):")
        logger.info(f"  Per-fold best epochs: {cv_best_epochs}")
        logger.info(f"  Mean: {mean_best_epoch} ± {cv_epochs_info['std_best_epoch']:.1f}")
        logger.info(f"  Range: [{cv_epochs_info['min_best_epoch']}, {cv_epochs_info['max_best_epoch']}]")
        logger.info(f"{'='*60}\n")
        
        return study.best_params, study, cv_epochs_info


def load_ig_importances(ig_ranking_dir: Path) -> Dict:
    """
    Load Integrated Gradients aggregated importances.
    
    Expects files:
    - tcga_ig_aggregated.csv (columns: gene_name, ig_mean, ig_std, rank, ...)
    - orien_ig_aggregated.csv (columns: gene_name, ig_mean, ig_std, rank, ...)
    
    Args:
        ig_ranking_dir: Directory containing aggregated IG files
        
    Returns:
        Dict with 'tcga_importances', 'orien_importances', 'gene_names'
    """
    logger.info("="*60)
    logger.info("Loading Integrated Gradients Aggregated Importances")
    logger.info("="*60)
    
    tcga_file = ig_ranking_dir / "tcga_ig_aggregated.csv"
    orien_file = ig_ranking_dir / "orien_ig_aggregated.csv"
    
    if not tcga_file.exists():
        raise FileNotFoundError(
            f"TCGA IG aggregated file not found: {tcga_file}\n"
            f"Please run aggregate_ig_score_ranking.py first."
        )
    
    if not orien_file.exists():
        raise FileNotFoundError(
            f"ORIEN IG aggregated file not found: {orien_file}\n"
            f"Please run aggregate_ig_score_ranking.py first."
        )
    
    # Load TCGA
    tcga_df = pd.read_csv(tcga_file)
    tcga_df = tcga_df.set_index('gene')  # Column name is 'gene' in aggregated files
    
    # Load ORIEN
    orien_df = pd.read_csv(orien_file)
    orien_df = orien_df.set_index('gene')  # Column name is 'gene' in aggregated files
    
    logger.info(f"Loaded TCGA: {len(tcga_df)} genes")
    logger.info(f"Loaded ORIEN: {len(orien_df)} genes")
    
    # Verify gene sets match
    common_genes = sorted(list(set(tcga_df.index) & set(orien_df.index)))
    if len(common_genes) != len(tcga_df) or len(common_genes) != len(orien_df):
        logger.warning(f"Gene mismatch: TCGA={len(tcga_df)}, ORIEN={len(orien_df)}, Common={len(common_genes)}")
        # Filter to common genes
        tcga_df = tcga_df.loc[common_genes]
        orien_df = orien_df.loc[common_genes]
    
    # Sort by IG importance (descending)
    tcga_importances = tcga_df['ig_mean'].sort_values(ascending=False)
    orien_importances = orien_df['ig_mean'].sort_values(ascending=False)
    
    logger.info(f"\nTCGA top 5 (by IG):")
    for i, (gene, score) in enumerate(tcga_importances.head().items()):
        logger.info(f"  {i+1}. {gene}: {score:.6f}")
    
    logger.info(f"\nORIEN top 5 (by IG):")
    for i, (gene, score) in enumerate(orien_importances.head().items()):
        logger.info(f"  {i+1}. {gene}: {score:.6f}")
    
    return {
        'tcga_importances': tcga_importances,
        'orien_importances': orien_importances,
        'gene_names': common_genes
    }


def select_consensus_genes_at_k(k: int, importances: Dict) -> Tuple[List[str], Dict]:
    """Select top-k genes and find consensus."""
    logger.info(f"\nk={k} Gene Selection (IG-based):")
    
    tcga_top_k = importances['tcga_importances'].head(k).index.tolist()
    orien_top_k = importances['orien_importances'].head(k).index.tolist()
    
    consensus_genes = sorted(list(set(tcga_top_k) & set(orien_top_k)))
    
    overlap_pct = len(consensus_genes) / k * 100
    random_overlap_pct = (k / len(importances['gene_names'])) * 100
    
    gene_info = {
        'k': k,
        'tcga_top_k': tcga_top_k,
        'orien_top_k': orien_top_k,
        'consensus_genes': consensus_genes,
        'm': len(consensus_genes),
        'overlap_pct': overlap_pct,
        'random_overlap_pct': random_overlap_pct,
        'importance_method': 'integrated_gradients'
    }
    
    logger.info(f"  TCGA top-{k}: {len(tcga_top_k)} genes")
    logger.info(f"  ORIEN top-{k}: {len(orien_top_k)} genes")
    logger.info(f"  Consensus: {len(consensus_genes)} genes ({overlap_pct:.1f}% overlap)")
    logger.info(f"  Random expectation: {random_overlap_pct:.1f}%")
    logger.info(f"  Enrichment: {overlap_pct / random_overlap_pct:.2f}x over random")
    
    return consensus_genes, gene_info


def tune_for_k_value(
    consensus_genes: List[str],
    k: int,
    cohort: str,
    output_dir: Path,
    data_dir: Path,
    n_trials: int = 50
) -> Dict:
    """Run hyperparameter tuning for a specific k-value and cohort."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Hyperparameter Tuning: {cohort.upper()} (k={k}, m={len(consensus_genes)}, IG-based)")
    logger.info(f"{'='*60}")
    
    # Create output directory
    tuning_dir = output_dir / f"k{k:03d}" / "hyperparameter_tuning" / cohort
    tuning_dir.mkdir(parents=True, exist_ok=True)
    
    # Save consensus genes temporarily
    temp_gene_file = data_dir / 'raw' / f'temp_consensus_ig_k{k}_{cohort}.txt'
    with open(temp_gene_file, 'w') as f:
        f.write('\n'.join(consensus_genes))
    
    try:
        # Load configuration
        with open('config/default_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Update config to use consensus genes
        config['data']['use_consensus_genes'] = True
        config['data']['consensus_gene_file'] = str(temp_gene_file)
        config['data']['use_common_genes'] = False
        config['data']['min_variance_percentile'] = 0
        
        # Load RAW data
        if cohort.lower() == 'tcga':
            expr_raw = pd.read_csv("data/raw/tcga_batch_corrected_2sv.csv", index_col=0)
            surv = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
        else:
            expr_raw = pd.read_csv("data/raw/orien_batch_corrected.csv", index_col=0)
            surv = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
        
        # Filter to consensus genes
        available_genes = [g for g in consensus_genes if g in expr_raw.index]
        if len(available_genes) < len(consensus_genes):
            logger.warning(f"Only {len(available_genes)}/{len(consensus_genes)} consensus genes available")
        
        expr_raw = expr_raw.loc[available_genes]
        logger.info(f"Loaded: {len(expr_raw)} genes × {expr_raw.shape[1]} samples")
        
        # Create tuner
        tuner = LeakageFreeKSelectionTuner(
            train_expr_raw=expr_raw,
            train_surv=surv,
            config=config,
            cohort_name=cohort,
            k_value=k
        )
        
        # Run optimization (returns cv_epochs_info for Option 2)
        best_params, study, cv_epochs_info = tuner.optimize(n_trials=n_trials)
        
        # Save results including CV-derived epochs
        results = {
            'cohort': cohort,
            'k': k,
            'm': len(available_genes),
            'best_params': best_params,
            'best_cv_cindex': study.best_value,
            'n_trials': n_trials,
            'importance_method': 'integrated_gradients',
            'cv_epochs_info': cv_epochs_info  # Option 2: CV-derived epochs
        }
        
        with open(tuning_dir / 'best_params.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        study.trials_dataframe().to_csv(tuning_dir / 'trials.csv', index=False)
        
        summary = {
            'best_value': study.best_value,
            'best_params': best_params,
            'n_trials': len(study.trials),
            'cv_epochs_info': cv_epochs_info
        }
        
        with open(tuning_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to: {tuning_dir}")
        
        return results
        
    finally:
        # Clean up temp file
        if temp_gene_file.exists():
            temp_gene_file.unlink()


def train_and_test_direction(
    source_cohort: str,
    target_cohort: str,
    source_params: Dict,
    consensus_genes: List[str],
    data_dir: Path,
    config: dict,
    cv_derived_epochs: int = None,  # Option 2: CV-derived epochs
    device: str = 'cuda'
) -> Dict:
    """
    Train on source (full cohort) → Test on target (full cohort).
    
    Uses CV-derived epochs (Option 2) for fair comparison with Cox elastic net:
    - Cox: Uses full data with built-in CV for lambda selection
    - DeepSurv: Uses full data with CV-derived epoch count
    
    This ensures both methods use 100% of training data while preventing overfitting
    through principled stopping criteria.
    
    Reference: Katzman et al. (2018) DeepSurv - fair comparison methodology
    """
    logger.info(f"\n--- Training: {source_cohort.upper()} → Testing: {target_cohort.upper()} ---")
    
    # Parse hyperparameters from source tuning
    best_params = source_params['best_params']
    
    # Get CV-derived epochs if available
    if cv_derived_epochs is None:
        cv_epochs_info = source_params.get('cv_epochs_info', {})
        cv_derived_epochs = cv_epochs_info.get('mean_best_epoch', 100)
        logger.info(f"Using CV-derived epochs from source tuning: {cv_derived_epochs}")
    else:
        logger.info(f"Using provided CV-derived epochs: {cv_derived_epochs}")
    
    # Parse architecture
    if 'architecture_2layer' in best_params:
        hidden_sizes = [int(x) for x in best_params['architecture_2layer'].split('-')]
    elif 'architecture_3layer' in best_params:
        hidden_sizes = [int(x) for x in best_params['architecture_3layer'].split('-')]
    elif 'layer1_size' in best_params:
        hidden_sizes = [best_params['layer1_size']]
    else:
        raise ValueError(f"Cannot parse architecture from {best_params}")
    
    dropout = best_params['dropout']
    learning_rate = best_params['learning_rate']
    alpha = best_params['alpha']
    l1_ratio = best_params['l1_ratio']
    batch_size = best_params['batch_size']
    activation = best_params['activation']
    batch_norm = best_params['batch_norm']
    weight_init = best_params.get('weight_init', 'kaiming_normal')
    
    logger.info(f"Using architecture: {hidden_sizes}")
    
    # Load RAW data for both cohorts
    tcga_expr_raw = pd.read_csv(data_dir / "raw" / "tcga_batch_corrected_2sv.csv", index_col=0)
    orien_expr_raw = pd.read_csv(data_dir / "raw" / "orien_batch_corrected.csv", index_col=0)
    tcga_surv = pd.read_csv(data_dir / "processed" / "surv_tcga_harmonized.csv", index_col=0)
    orien_surv = pd.read_csv(data_dir / "processed" / "surv_orien_harmonized.csv", index_col=0)
    
    # Filter to consensus genes (handle missing genes)
    available_tcga = [g for g in consensus_genes if g in tcga_expr_raw.index]
    available_orien = [g for g in consensus_genes if g in orien_expr_raw.index]
    available_genes = sorted(list(set(available_tcga) & set(available_orien)))
    
    tcga_expr_raw = tcga_expr_raw.loc[available_genes]
    orien_expr_raw = orien_expr_raw.loc[available_genes]
    
    logger.info(f"TCGA: {tcga_expr_raw.shape[0]} genes × {tcga_expr_raw.shape[1]} samples")
    logger.info(f"ORIEN: {orien_expr_raw.shape[0]} genes × {orien_expr_raw.shape[1]} samples")
    
    # Preprocess BOTH cohorts
    preprocessor = GeneExpressionPreprocessor(config)
    
    if source_cohort.lower() == 'tcga':
        source_processed = preprocessor.fit_transform_single_cohort(
            tcga_expr_raw,
            cohort_name='TCGA'
        )
        target_processed = preprocessor.transform_single_cohort(orien_expr_raw)
        source_surv = tcga_surv
        target_surv = orien_surv
    else:  # source is ORIEN
        source_processed = preprocessor.fit_transform_single_cohort(
            orien_expr_raw,
            cohort_name='ORIEN'
        )
        target_processed = preprocessor.transform_single_cohort(tcga_expr_raw)
        source_surv = orien_surv
        target_surv = tcga_surv
    
    # Create datasets
    source_dataset = SurvivalDataset(source_processed, source_surv)
    target_dataset = SurvivalDataset(target_processed, target_surv)
    
    # Create dataloaders
    source_events = source_surv['event'].values
    source_sampler = StratifiedBatchSampler(
        events=source_events,
        batch_size=batch_size,
        min_events_per_batch=2,
        shuffle=True
    )
    
    source_loader = DataLoader(
        source_dataset,
        batch_sampler=source_sampler,
        num_workers=0
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    n_features = source_processed.shape[0]
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
    
    logger.info(f"Model created: {n_features} → {hidden_sizes} → 1")
    
    # Create trainer
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device
    )
    
    # =======================================================================
    # OPTION 2: CV-DERIVED EPOCHS
    # Train on source (full cohort) for CV-derived number of epochs
    # This matches Cox elastic net's use of full data with principled stopping
    # =======================================================================
    logger.info(f"Training on {source_cohort.upper()} (full cohort: {len(source_dataset)} samples)...")
    logger.info(f"Using CV-derived epochs: {cv_derived_epochs} (Option 2 for fair Cox comparison)")
    
    history = trainer.fit(
        train_loader=source_loader,
        valid_loader=None,           # No validation set - use full data
        n_epochs=cv_derived_epochs,  # CV-derived epochs instead of fixed 100
        early_stopping_patience=None,
        verbose=False
    )
    
    # Get training performance
    _, _, _, train_cindex = trainer.evaluate(source_loader)
    logger.info(f"  Training C-index on {source_cohort.upper()}: {train_cindex:.4f}")
    
    # Test on target
    logger.info(f"Testing on {target_cohort.upper()} (full cohort: {len(target_dataset)} samples)...")
    _, _, _, test_cindex = trainer.evaluate(target_loader)
    logger.info(f"  Test C-index on {target_cohort.upper()}: {test_cindex:.4f}")
    
    return {
        'source': source_cohort,
        'target': target_cohort,
        'train_cindex': train_cindex,
        'test_cindex': test_cindex,
        'architecture': hidden_sizes,
        'n_source_samples': len(source_dataset),
        'n_target_samples': len(target_dataset),
        'cv_derived_epochs': cv_derived_epochs  # Record epochs used
    }


def cross_cohort_validation(
    consensus_genes: List[str],
    tcga_params: Dict,
    orien_params: Dict,
    k: int,
    output_dir: Path,
    data_dir: Path
) -> Dict:
    """
    Cross-cohort validation using optimal hyperparameters.
    
    Uses Option 2 (CV-derived epochs) for fair comparison with Cox elastic net:
    - Both methods use 100% of source training data
    - DeepSurv stops at epoch count derived from CV (no holdout required)
    - Cox stops via built-in CV for lambda selection
    
    Performs bidirectional validation:
    1. Train ORIEN → Test TCGA (using ORIEN's CV-derived epochs)
    2. Train TCGA → Test ORIEN (using TCGA's CV-derived epochs)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Cross-Cohort Validation (k={k}, m={len(consensus_genes)}, IG-based)")
    logger.info(f"Using Option 2: CV-derived epochs for fair Cox comparison")
    logger.info(f"{'='*60}")
    
    # Log CV-derived epochs being used
    orien_cv_epochs = orien_params.get('cv_epochs_info', {}).get('mean_best_epoch', 100)
    tcga_cv_epochs = tcga_params.get('cv_epochs_info', {}).get('mean_best_epoch', 100)
    logger.info(f"  ORIEN CV-derived epochs: {orien_cv_epochs}")
    logger.info(f"  TCGA CV-derived epochs: {tcga_cv_epochs}")
    
    # Create output directory
    validation_dir = output_dir / f"k{k:03d}" / "cross_cohort_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config
    config['data']['use_consensus_genes'] = False
    config['data']['min_variance_percentile'] = 0
    config['data']['standardize'] = True
    
    # Direction 1: ORIEN → TCGA (use ORIEN's CV-derived epochs)
    logger.info("\n" + "="*60)
    logger.info("Direction 1: ORIEN → TCGA")
    logger.info(f"Training epochs: {orien_cv_epochs} (CV-derived from ORIEN tuning)")
    logger.info("="*60)
    
    o2t_results = train_and_test_direction(
        source_cohort='orien',
        target_cohort='tcga',
        source_params=orien_params,
        consensus_genes=consensus_genes,
        data_dir=data_dir,
        config=config
        # cv_derived_epochs extracted from source_params inside function
    )
    
    # Direction 2: TCGA → ORIEN (use TCGA's CV-derived epochs)
    logger.info("\n" + "="*60)
    logger.info("Direction 2: TCGA → ORIEN")
    logger.info(f"Training epochs: {tcga_cv_epochs} (CV-derived from TCGA tuning)")
    logger.info("="*60)
    
    t2o_results = train_and_test_direction(
        source_cohort='tcga',
        target_cohort='orien',
        source_params=tcga_params,
        consensus_genes=consensus_genes,
        data_dir=data_dir,
        config=config
        # cv_derived_epochs extracted from source_params inside function
    )
    
    # Calculate bidirectional statistics
    o2t_cindex = o2t_results['test_cindex']
    t2o_cindex = t2o_results['test_cindex']
    mean_cindex = (o2t_cindex + t2o_cindex) / 2
    
    # Compile results (include CV-derived epochs for transparency)
    summary = {
        'k': k,
        'm': len(consensus_genes),
        'importance_method': 'integrated_gradients',
        'stopping_method': 'cv_derived_epochs',  # Document the method used
        'orien_to_tcga': {
            'train_cindex': o2t_results['train_cindex'],
            'test_cindex': o2t_results['test_cindex'],
            'architecture': o2t_results['architecture'],
            'cv_derived_epochs': o2t_results.get('cv_derived_epochs', orien_cv_epochs)
        },
        'tcga_to_orien': {
            'train_cindex': t2o_results['train_cindex'],
            'test_cindex': t2o_results['test_cindex'],
            'architecture': t2o_results['architecture'],
            'cv_derived_epochs': t2o_results.get('cv_derived_epochs', tcga_cv_epochs)
        },
        'orien_to_tcga_cindex': o2t_cindex,
        'tcga_to_orien_cindex': t2o_cindex,
        'mean_bidirectional_cindex': mean_cindex,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    with open(validation_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save consensus genes
    with open(validation_dir / 'consensus_genes.txt', 'w') as f:
        f.write('\n'.join(consensus_genes))
    
    logger.info(f"\n{'='*60}")
    logger.info("Cross-Cohort Validation Results:")
    logger.info(f"{'='*60}")
    logger.info(f"  ORIEN → TCGA (epochs={orien_cv_epochs}):")
    logger.info(f"    Train C-index: {o2t_results['train_cindex']:.4f}")
    logger.info(f"    Test C-index:  {o2t_cindex:.4f}")
    logger.info(f"  TCGA → ORIEN (epochs={tcga_cv_epochs}):")
    logger.info(f"    Train C-index: {t2o_results['train_cindex']:.4f}")
    logger.info(f"    Test C-index:  {t2o_cindex:.4f}")
    logger.info(f"  Mean Bidirectional: {mean_cindex:.4f}")
    logger.info(f"{'='*60}\n")
    logger.info(f"Results saved to: {validation_dir}")
    
    return summary


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='K-Selection with Per-K Hyperparameter Tuning using Integrated Gradients'
    )
    parser.add_argument('--k_values', nargs='+', type=int, 
                       default=[80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180],
                       help='K values to test')
    parser.add_argument('--output_dir', type=str,
                       default='results_v2/02b_biomarker_discovery_ig/k_selection_with_tuning',
                       help='Output directory')
    parser.add_argument('--ig_ranking_dir', type=str,
                       default='results_v2/06_importance_methods/aggregated',
                       help='Directory containing aggregated IG rankings')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Optuna trials per cohort')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    ig_ranking_dir = Path(args.ig_ranking_dir)
    data_dir = Path(args.data_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("STEP 2 (IG): K-SELECTION WITH INTEGRATED GRADIENTS IMPORTANCE")
    logger.info("="*80)
    logger.info(f"K values: {args.k_values}")
    logger.info(f"IG ranking directory: {ig_ranking_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Optuna trials: {args.n_trials}")
    logger.info("="*80)
    
    # Load IG importances
    importances = load_ig_importances(ig_ranking_dir)
    
    all_results = []
    
    # Process each k-value
    for k in args.k_values:
        logger.info("\n" + "="*80)
        logger.info(f"PROCESSING k={k}")
        logger.info("="*80)
        
        try:
            # 1. Select consensus genes
            consensus_genes, gene_info = select_consensus_genes_at_k(k, importances)
            
            # Save gene lists
            gene_dir = output_dir / f"k{k:03d}" / "consensus_genes"
            gene_dir.mkdir(parents=True, exist_ok=True)
            
            with open(gene_dir / 'consensus_genes.txt', 'w') as f:
                f.write('\n'.join(consensus_genes))
            
            with open(gene_dir / 'gene_info.json', 'w') as f:
                json.dump(gene_info, f, indent=2)
            
            # 2. Hyperparameter tuning for TCGA
            tcga_results = tune_for_k_value(
                consensus_genes, k, 'tcga', output_dir, data_dir, args.n_trials
            )
            
            # 3. Hyperparameter tuning for ORIEN
            orien_results = tune_for_k_value(
                consensus_genes, k, 'orien', output_dir, data_dir, args.n_trials
            )
            
            # 4. Cross-cohort validation
            validation_results = cross_cohort_validation(
                consensus_genes, tcga_results, orien_results, k, output_dir, data_dir
            )
            
            # Compile results
            k_results = {
                'k': k,
                'm': len(consensus_genes),
                'overlap_pct': gene_info['overlap_pct'],
                'tcga_cv_cindex': tcga_results['best_cv_cindex'],
                'orien_cv_cindex': orien_results['best_cv_cindex'],
                'orien_to_tcga_cindex': validation_results['orien_to_tcga_cindex'],
                'tcga_to_orien_cindex': validation_results['tcga_to_orien_cindex'],
                'mean_bidirectional_cindex': validation_results['mean_bidirectional_cindex'],
                'importance_method': 'integrated_gradients'
            }
            
            all_results.append(k_results)
            
        except Exception as e:
            logger.error(f"Error processing k={k}: {e}", exc_info=True)
            continue
    
    # Save summary
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(summary_dir / 'k_selection_summary.csv', index=False)
    
    # Find optimal k
    if len(summary_df) > 0:
        optimal_idx = summary_df['mean_bidirectional_cindex'].idxmax()
        optimal_k = summary_df.loc[optimal_idx, 'k']
        optimal_cindex = summary_df.loc[optimal_idx, 'mean_bidirectional_cindex']
        optimal_m = summary_df.loc[optimal_idx, 'm']
        
        optimal_info = {
            'optimal_k': int(optimal_k),
            'optimal_m': int(optimal_m),
            'optimal_mean_bidirectional_cindex': float(optimal_cindex),
            'selection_criterion': 'max_mean_bidirectional_cindex',
            'importance_method': 'integrated_gradients',
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_dir / 'optimal_k_recommendation.json', 'w') as f:
            json.dump(optimal_info, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("K-SELECTION SUMMARY (IG-based)")
        logger.info("="*80)
        logger.info(summary_df.to_string(index=False))
        logger.info(f"\nOPTIMAL: k={optimal_k}, m={optimal_m}, C-index={optimal_cindex:.4f}")
        logger.info("="*80)
    
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
