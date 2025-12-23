"""
Step 2 (IG + Sign Filter): K-Selection with Sign-Consistent Genes Only
=======================================================================

This script is a modified version of step2_tune_and_validate_k_ig.py that:
1. Filters to 141 sign-consistent genes BEFORE ranking
2. Uses IG importance for gene ranking within the filtered pool
3. Outputs to results_v2/02c_biomarker_discovery_ig_signfilter/

Key Methodological Difference:
- Original (02b): 308 genes → top-k selection → consensus
- This (02c): 141 sign-consistent genes → top-k selection → consensus

Rationale for Sign Filtering:
- Genes with opposite effect directions across cohorts are biologically incoherent
- Sign consistency ensures:
  1. Interpretable biomarkers (consistent risk/protective direction)
  2. Better transfer learning (aligns conditional distributions)
  3. Clinically meaningful gene panels

Reference:
- Zhuang et al. (2020) "A Comprehensive Survey on Transfer Learning" - domain alignment
- Bernau et al. (2014) Bioinformatics - Cross-study validation principles

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

Reference: Katzman et al. (2018) DeepSurv - methodology for fair comparison
=============================================================================

For each k-value:
    1. Extract top-k genes from sign-filtered IG rankings (TCGA and ORIEN)
    2. Find consensus genes (intersection)
    3. Hyperparameter tuning on TCGA with proper CV preprocessing
    4. Hyperparameter tuning on ORIEN with proper CV preprocessing
    5. Cross-cohort validation with CV-derived epochs (Option 2)

Based on:
- Sundararajan et al. (2017, ICML) - Integrated Gradients
- Bergstra & Bengio (2012, JMLR) - Hyperparameter optimization
- Bernau et al. (2014, Bioinformatics) - Cross-study validation
- Harrell (2001) - EPV guidelines for survival models
- van der Ploeg et al. (2014, BMC Med Res Methodol) - EPV in prediction models

Author: Phuong (modified for sign-consistent gene filtering)
Date: December 2025
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
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


# =============================================================================
# CONFIGURATION - Sign-Filtered Version
# =============================================================================
DEFAULT_CONFIG = {
    'sign_consistent_genes': 'data/processed/sign_consistent_genes_141.txt',
    'ig_ranking_dir': 'results_v2/06_importance_methods/aggregated',
    'output_dir': 'results_v2/02c_biomarker_discovery_ig_signfilter/k_selection_with_tuning',
    'data_dir': 'data',
    'k_values': [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],  # Based on pre-analysis
    'n_trials': 50,
}


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
        # - Consensus genes (m): varies with k
        # - Architecture should scale with m, not exceed event count
        # =================================================================
        
        m = self.n_genes_raw  # Number of consensus genes (input features)
        
        if self.n_samples < 500:  # TCGA (~153 events)
            # 1-layer architectures, scaled by m
            if m <= 40:
                layer_options = [16, 32, 48]
            elif m <= 60:
                layer_options = [32, 48, 64]
            elif m <= 80:
                layer_options = [48, 64, 96]
            else:  # m > 80
                layer_options = [64, 96, 128]
            
            layer1_size = trial.suggest_categorical('layer1_size', layer_options)
            hidden_sizes = [layer1_size]
            
            # Stronger regularization for small sample
            alpha = trial.suggest_float('alpha', 1e-4, 5e-3, log=True)
            
        else:  # ORIEN (~450 events)
            # More flexibility
            if m <= 40:
                layer_options = [32, 48, 64]
                layer1_size = trial.suggest_categorical('layer1_size', layer_options)
                hidden_sizes = [layer1_size]
            elif m <= 70:
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
            else:  # m > 70
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
            alpha = trial.suggest_float('alpha', 5e-5, 1e-3, log=True)
        
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
        cv_best_epochs = []
        
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
                    n_epochs=200,
                    early_stopping_patience=30,
                    verbose=False
                )
                
                best_cindex = max(history['valid_c_index'])
                cv_scores.append(best_cindex)
                
                # Capture best_epoch for CV-derived stopping (Option 2)
                fold_best_epoch = history.get('best_epoch', len(history['train_loss']))
                cv_best_epochs.append(fold_best_epoch)
                
            except Exception as e:
                logger.warning(f"    Fold {fold+1} failed: {e}")
                cv_scores.append(0.5)
                cv_best_epochs.append(50)
                continue
            
            # Report for pruning
            trial.report(np.mean(cv_scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        mean_cindex = np.mean(cv_scores)
        std_cindex = np.std(cv_scores)
        
        # Store CV best epochs as trial attribute (for Option 2)
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
        study_name = f"k{self.k_value}_{self.cohort_name}_IG_signfilter"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"HYPERPARAMETER OPTIMIZATION (k={self.k_value}, {self.cohort_name}, IG+SignFilter)")
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


def load_sign_consistent_genes(filepath: str) -> List[str]:
    """
    Load the list of sign-consistent genes.
    
    These are genes that have consistent effect direction (risk or protective)
    across both TCGA and ORIEN cohorts.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Sign-consistent genes file not found: {filepath}\n"
            f"Please run sign analysis first to generate this file."
        )
    
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(genes)} sign-consistent genes from {filepath}")
    return genes


def load_ig_importances_signfiltered(
    ig_ranking_dir: Path,
    sign_consistent_genes: List[str]
) -> Dict:
    """
    Load IG aggregated importances and filter to sign-consistent genes.
    
    This is the KEY MODIFICATION from the original script:
    - Original: Uses all 308 genes
    - This version: Filters to 141 sign-consistent genes first
    
    The ranking order (by ig_mean) is preserved within the filtered set.
    
    Args:
        ig_ranking_dir: Directory containing tcga_ig_aggregated.csv and orien_ig_aggregated.csv
        sign_consistent_genes: List of 141 sign-consistent genes
        
    Returns:
        Dict with 'tcga_importances', 'orien_importances', 'gene_names', 'n_original', 'n_filtered'
    """
    logger.info("="*60)
    logger.info("Loading IG Importances (SIGN-FILTERED)")
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
    
    # Load original rankings (308 genes)
    tcga_df = pd.read_csv(tcga_file)
    tcga_df = tcga_df.set_index('gene')
    
    orien_df = pd.read_csv(orien_file)
    orien_df = orien_df.set_index('gene')
    
    n_original = len(tcga_df)
    logger.info(f"Original gene count: {n_original}")
    
    # =========================================================================
    # KEY STEP: Filter to sign-consistent genes
    # =========================================================================
    sign_gene_set = set(sign_consistent_genes)
    
    tcga_filtered = tcga_df.loc[tcga_df.index.isin(sign_gene_set)]
    orien_filtered = orien_df.loc[orien_df.index.isin(sign_gene_set)]
    
    n_filtered = len(tcga_filtered)
    logger.info(f"After sign filtering: {n_filtered} genes")
    
    # Verify all sign-consistent genes are present
    missing_tcga = sign_gene_set - set(tcga_filtered.index)
    missing_orien = sign_gene_set - set(orien_filtered.index)
    
    if missing_tcga:
        logger.warning(f"Missing {len(missing_tcga)} sign-consistent genes from TCGA rankings")
    if missing_orien:
        logger.warning(f"Missing {len(missing_orien)} sign-consistent genes from ORIEN rankings")
    
    # Get common genes between filtered sets
    common_genes = sorted(list(set(tcga_filtered.index) & set(orien_filtered.index)))
    
    if len(common_genes) < n_filtered:
        logger.warning(f"Gene mismatch after filtering: {len(common_genes)} common genes")
        tcga_filtered = tcga_filtered.loc[common_genes]
        orien_filtered = orien_filtered.loc[common_genes]
    
    # Sort by IG importance (descending) - preserves ranking within filtered set
    tcga_importances = tcga_filtered['ig_mean'].sort_values(ascending=False)
    orien_importances = orien_filtered['ig_mean'].sort_values(ascending=False)
    
    logger.info(f"\nSign-filtered TCGA top 5 (by IG):")
    for i, (gene, score) in enumerate(tcga_importances.head().items()):
        logger.info(f"  {i+1}. {gene}: {score:.6f}")
    
    logger.info(f"\nSign-filtered ORIEN top 5 (by IG):")
    for i, (gene, score) in enumerate(orien_importances.head().items()):
        logger.info(f"  {i+1}. {gene}: {score:.6f}")
    
    logger.info(f"\nFiltering summary:")
    logger.info(f"  Original genes: {n_original}")
    logger.info(f"  Sign-consistent: {len(sign_consistent_genes)}")
    logger.info(f"  Available for ranking: {len(common_genes)}")
    
    return {
        'tcga_importances': tcga_importances,
        'orien_importances': orien_importances,
        'gene_names': common_genes,
        'n_original': n_original,
        'n_filtered': len(common_genes)
    }


def select_consensus_genes_at_k(k: int, importances: Dict) -> Tuple[List[str], Dict]:
    """Select top-k genes and find consensus."""
    n_available = len(importances['gene_names'])
    
    # Handle k > available genes
    k_actual = min(k, n_available)
    
    if k_actual < k:
        logger.warning(f"Requested k={k} but only {n_available} genes available. Using k={k_actual}")
    
    logger.info(f"\nk={k_actual} Gene Selection (IG + Sign-Filtered):")
    
    tcga_top_k = importances['tcga_importances'].head(k_actual).index.tolist()
    orien_top_k = importances['orien_importances'].head(k_actual).index.tolist()
    
    consensus_genes = sorted(list(set(tcga_top_k) & set(orien_top_k)))
    
    overlap_pct = len(consensus_genes) / k_actual * 100
    random_overlap_pct = (k_actual / n_available) * 100
    enrichment = overlap_pct / random_overlap_pct if random_overlap_pct > 0 else 0
    
    gene_info = {
        'k': k,
        'k_actual': k_actual,
        'tcga_top_k': tcga_top_k,
        'orien_top_k': orien_top_k,
        'consensus_genes': consensus_genes,
        'm': len(consensus_genes),
        'overlap_pct': overlap_pct,
        'random_overlap_pct': random_overlap_pct,
        'enrichment': enrichment,
        'importance_method': 'integrated_gradients',
        'gene_pool': 'sign_consistent_141'
    }
    
    logger.info(f"  TCGA top-{k_actual}: {len(tcga_top_k)} genes")
    logger.info(f"  ORIEN top-{k_actual}: {len(orien_top_k)} genes")
    logger.info(f"  Consensus: {len(consensus_genes)} genes ({overlap_pct:.1f}% overlap)")
    logger.info(f"  Random expectation: {random_overlap_pct:.1f}%")
    logger.info(f"  Enrichment: {enrichment:.2f}x over random")
    
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
    logger.info(f"Hyperparameter Tuning: {cohort.upper()} (k={k}, m={len(consensus_genes)}, IG+SignFilter)")
    logger.info(f"{'='*60}")
    
    # Create output directory
    tuning_dir = output_dir / f"k{k:03d}" / "hyperparameter_tuning" / cohort
    tuning_dir.mkdir(parents=True, exist_ok=True)
    
    # Save consensus genes temporarily
    temp_gene_file = data_dir / 'raw' / f'temp_consensus_ig_signfilter_k{k}_{cohort}.txt'
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
            expr_raw = pd.read_csv(data_dir / "raw" / "tcga_batch_corrected_2sv.csv", index_col=0)
            surv = pd.read_csv(data_dir / "processed" / "surv_tcga_harmonized.csv", index_col=0)
        else:
            expr_raw = pd.read_csv(data_dir / "raw" / "orien_batch_corrected.csv", index_col=0)
            surv = pd.read_csv(data_dir / "processed" / "surv_orien_harmonized.csv", index_col=0)
        
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
            'gene_pool': 'sign_consistent_141',
            'cv_epochs_info': cv_epochs_info
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


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='K-Selection with Sign-Consistent Genes (IG-based)'
    )
    parser.add_argument('--k_values', nargs='+', type=int, 
                       default=DEFAULT_CONFIG['k_values'],
                       help='K values to test')
    parser.add_argument('--output_dir', type=str,
                       default=DEFAULT_CONFIG['output_dir'],
                       help='Output directory')
    parser.add_argument('--ig_ranking_dir', type=str,
                       default=DEFAULT_CONFIG['ig_ranking_dir'],
                       help='Directory containing aggregated IG rankings')
    parser.add_argument('--sign_genes', type=str,
                       default=DEFAULT_CONFIG['sign_consistent_genes'],
                       help='Path to sign-consistent genes file')
    parser.add_argument('--data_dir', type=str, 
                       default=DEFAULT_CONFIG['data_dir'],
                       help='Data directory')
    parser.add_argument('--n_trials', type=int, 
                       default=DEFAULT_CONFIG['n_trials'],
                       help='Number of Optuna trials per cohort')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    ig_ranking_dir = Path(args.ig_ranking_dir)
    data_dir = Path(args.data_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("STEP 2 (IG + SIGN FILTER): K-SELECTION WITH SIGN-CONSISTENT GENES")
    logger.info("="*80)
    logger.info(f"Sign-consistent genes file: {args.sign_genes}")
    logger.info(f"IG ranking directory: {ig_ranking_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"K values: {args.k_values}")
    logger.info(f"Optuna trials: {args.n_trials}")
    logger.info("="*80)
    
    # Load sign-consistent genes
    sign_genes = load_sign_consistent_genes(args.sign_genes)
    
    # Load IG importances (filtered to sign-consistent genes)
    importances = load_ig_importances_signfiltered(ig_ranking_dir, sign_genes)
    
    # Save metadata about the gene pool
    metadata = {
        'gene_pool': 'sign_consistent',
        'n_sign_consistent': len(sign_genes),
        'n_available': importances['n_filtered'],
        'n_original': importances['n_original'],
        'sign_genes_file': args.sign_genes,
        'ig_ranking_dir': str(ig_ranking_dir),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'gene_pool_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    all_results = []
    
    # Process each k-value
    for k in args.k_values:
        logger.info("\n" + "="*80)
        logger.info(f"PROCESSING k={k}")
        logger.info("="*80)
        
        try:
            # 1. Select consensus genes
            consensus_genes, gene_info = select_consensus_genes_at_k(k, importances)
            
            # Skip if too few consensus genes
            if len(consensus_genes) < 10:
                logger.warning(f"Skipping k={k}: only {len(consensus_genes)} consensus genes (minimum 10)")
                continue
            
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
            
            k_results = {
                'k': k,
                'm': len(consensus_genes),
                'overlap_pct': gene_info['overlap_pct'],
                'enrichment': gene_info['enrichment'],
                'tcga_cv_cindex': tcga_results['best_cv_cindex'],
                'orien_cv_cindex': orien_results['best_cv_cindex'],
                'importance_method': 'integrated_gradients',
                'gene_pool': 'sign_consistent_141'
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
    
    if len(summary_df) > 0:
        logger.info("\n" + "="*80)
        logger.info("K-SELECTION TUNING COMPLETE (IG + Sign Filter)")
        logger.info("="*80)
        logger.info(summary_df.to_string(index=False))
        logger.info("\nNOTE: Run cross-cohort validation script to determine optimal k")
        logger.info("="*80)
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    # Print next steps
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("1. Run cross-cohort validation:")
    logger.info(f"   python run_step2b_crosscohort_validation_only_ig.py \\")
    logger.info(f"       --input_dir {output_dir} \\")
    logger.info(f"       --data_dir {data_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
