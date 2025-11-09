"""
Phase 1: Alpha Parameter Investigation - Fixed Grid Search
===========================================================

Research Question: Is weak alpha (0.0008) the root cause of regularization failure?

Hypothesis: Stronger alpha values will produce:
  - Higher sparsity (more genes with near-zero importance)
  - Lower C-index (regularization-performance trade-off)
  - More stable biomarker selection

Method: Fixed alpha grid search without Optuna optimization
Evidence base: 
  - Simon et al. (2011) "Regularization Paths for Cox Regression"
  - Zou & Hastie (2005) "Regularization and variable selection via elastic net"

Author: Phuong Nguyen
Date: 2024-11-06
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
from sklearn.model_selection import train_test_split


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



class AlphaInvestigator:
    """
    Systematic investigation of alpha parameter effects.
    
    Tests fixed alpha values to understand:
    1. Alpha-sparsity relationship
    2. Alpha-performance trade-off
    3. Gradient stability across alpha values
    """
    
    def __init__(self, config_path: str, output_dir: str, cohort: str = 'tcga'):
        """
        Initialize investigator.
        
        Args:
            config_path: Path to experiment config
            output_dir: Directory to save results
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set seeds for reproducibility
        self.seed = self.config['project']['seed']
        self._set_seeds()
        
        # Alpha values to test (logarithmically spaced)
        # Based on Cox elastic net literature (Simon et al. 2011)
        # self.alpha_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.lambda_values = [0.00005, 0.0001, 0.00025, 0.0005, 0.00075, 0.001]
        self.l1_ratio_values = [0.3, 0.5, 0.7, 0.9, 1.0]
        self.alpha_values = self.lambda_values
        self.cohort = cohort.lower()
        
        logger.info(f"Cohort for hyperparameter search: {self.cohort.upper()}")
        logger.info(f"Testing {len(self.lambda_values)} lambda values: {self.lambda_values}")
        logger.info(f"Testing {len(self.l1_ratio_values)} l1_ratio values: {self.l1_ratio_values}")
        logger.info(f"Total combinations: {len(self.lambda_values) * len(self.l1_ratio_values)}")
            
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
        """Load TCGA dataset for investigation."""
        logger.info("="*60)
        logger.info("Loading TCGA data...")
        
        data = load_dataset_from_config(self.config)
        
        if self.cohort == 'tcga':
            self.expr_data = data['tcga_expr']
            self.surv_data = data['surv_tcga']
        else:  # orien
            self.expr_data = data['orien_expr']
            self.surv_data = data['surv_orien']
        
        # Convert to tensors
        self.X = torch.FloatTensor(self.expr_data.T.values).to(self.device)
        self.T = torch.FloatTensor(self.surv_data['time'].values).to(self.device)
        self.E = torch.FloatTensor(self.surv_data['event'].values).to(self.device)

        self.n_features = self.X.shape[1]
        
        logger.info(f"Cohort samples: {self.X.shape[0]}")
        logger.info(f"Features (genes): {self.n_features}")
        logger.info(f"Events: {self.E.sum().item()}/{len(self.E)} ({100*self.E.mean().item():.1f}%)")
        logger.info("="*60)

    def train_with_fixed_params(self, lambda_val: float, l1_ratio: float, fold: int = 0) -> dict:
        """
        Train model with fixed lambda and l1_ratio values.
        
        Args:
            lambda_val: Lambda value (proximal operator strength)
            l1_ratio: L1 ratio (Group Lasso vs Ridge mix)
            fold: Cross-validation fold number
            
        Returns:
            dict with training results and metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with lambda={lambda_val:.5f}, l1_ratio={l1_ratio:.2f}")
        logger.info(f"{'='*60}")
        
        # Fixed hyperparameters
        config = {
            'n_features': self.n_features,
            'hidden_sizes': [256, 64],
            'dropout': 0.3,
            'activation': 'relu',
            'batch_norm': True,
            'alpha': lambda_val,  # This is just for model initialization
            'l1_ratio': l1_ratio,  # UPDATED: use the parameter
        }
        
        # Training parameters
        learning_rate = 0.000337
        num_epochs = 100
        batch_size = 32
        patience = 20
        
        # Create model
        model = ElasticDeepSurv(**config).to(self.device)
        
        # Create optimizer with specified lambda and l1_ratio
        optimizer = create_proximal_optimizer(
            model.parameters(),
            optimizer_type='fista',
            lr=learning_rate,
            alpha=0.1,  # Dummy value, not used
            l1_ratio=l1_ratio,  # UPDATED: use the parameter
            use_group_lasso=True,
            lambda_scale=lambda_val  # Direct lambda control
        )
        # Create enhanced stratification bins (event + time)
        strat_bins = create_survival_stratification_bins(
            self.T.cpu().numpy(),
            self.E.cpu().numpy(),
            n_time_bins=4
        )

        # Stratified split using enhanced bins (80/20 train/val)
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            np.arange(len(self.X)),
            test_size=0.2,
            stratify=strat_bins,  # Use enhanced stratification
            random_state=self.seed
        )

        X_train, X_val = self.X[train_idx], self.X[val_idx]
        T_train, T_val = self.T[train_idx], self.T[val_idx]
        E_train, E_val = self.E[train_idx], self.E[val_idx]

        # Log split statistics
        logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        logger.info(f"Event rates - Train: {E_train.mean().item():.1%}, Val: {E_val.mean().item():.1%}")

        # Create datasets for DataLoader
        train_dataset = TensorDataset(X_train, T_train, E_train)
        val_dataset = TensorDataset(X_val, T_val, E_val)

        # Create stratified batch sampler (matching production code)
        train_sampler = StratifiedBatchSampler(
            events=E_train.cpu().numpy(),
            batch_size=batch_size,
            shuffle=True
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # Training metrics storage
        train_losses = []
        val_c_indices = []
        gradient_norms = []

        best_c_index = 0
        patience_counter = 0
        best_state = None

        # Training loop with DataLoader
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            epoch_grad_norm = 0
            n_batches = 0
            
            for batch_data in train_loader:
                X_batch, T_batch, E_batch = batch_data
                
                # Forward pass
                risk_scores = model(X_batch)
                
                # Cox loss (elastic net penalty already in model)
                cox_criterion = CoxPHLoss()
                loss = cox_criterion(risk_scores, T_batch, E_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Track gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                epoch_grad_norm += total_norm
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / n_batches
            avg_grad_norm = epoch_grad_norm / n_batches
            train_losses.append(avg_loss)
            gradient_norms.append(avg_grad_norm)
            
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
            
            # Concatenate all validation batches
            all_val_risks = torch.cat(all_val_risks)
            all_val_T = torch.cat(all_val_T)
            all_val_E = torch.cat(all_val_E)
            
            val_c_index = concordance_index(
                all_val_T.cpu().numpy(),
                -all_val_risks.cpu().numpy().flatten(),
                all_val_E.cpu().numpy()
            )
            val_c_indices.append(val_c_index)
            
            if (epoch + 1) % 10 == 0:
                # Get first layer weights
                first_layer_weight = None
                for name, param in model.named_parameters():
                    if 'fc0.weight' in name:
                        first_layer_weight = param.data
                        break
                
                if first_layer_weight is not None:
                    with torch.no_grad():
                        # Gene-level norms (columns)
                        gene_norms = torch.norm(first_layer_weight, p=2, dim=0)
                        zero_genes = (gene_norms < 1e-4).sum().item()
                        active_genes = len(gene_norms) - zero_genes
                        
                        # More detailed diagnostics
                        logger.info(
                            f"  [Epoch {epoch+1}/{num_epochs}] "
                            f"Val C-index: {val_c_index:.4f} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"Grad: {avg_grad_norm:.4f}"
                        )
                        logger.info(
                            f"    Gene norms: "
                            f"min={gene_norms.min():.6f}, "
                            f"max={gene_norms.max():.6f}, "
                            f"mean={gene_norms.mean():.6f}, "
                            f"median={gene_norms.median():.6f}"
                        )
                        logger.info(
                            f"    Sparsity: {zero_genes}/{len(gene_norms)} genes zeroed "
                            f"({100*zero_genes/len(gene_norms):.1f}%) | "
                            f"Active: {active_genes}"
                        )
                else:
                    # Regular logging if first layer not found
                    logger.info(
                        f"  Epoch {epoch+1}/{num_epochs} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Val C-index: {val_c_index:.4f} | "
                        f"Grad norm: {avg_grad_norm:.4f}"
                    )
            
            # Early stopping
            if val_c_index > best_c_index:
                best_c_index = val_c_index
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        model.load_state_dict(best_state)
        
        # Extract feature importances
        model.eval()
        feature_importances = self._extract_feature_importance(model)
        
        # Calculate sparsity metrics
        sparsity_metrics = self._calculate_sparsity_metrics(feature_importances)
        
        # Final validation metrics
        with torch.no_grad():
            val_risks = model(X_val)
            final_c_index = concordance_index(
                T_val.cpu().numpy(),
                -val_risks.cpu().numpy().flatten(),
                E_val.cpu().numpy()
            )
        
        results = {
            'lambda': lambda_val,  # UPDATED: store lambda
            'l1_ratio': l1_ratio,
            'fold': fold,
            'config': config,
            'final_val_c_index': final_c_index,
            'best_val_c_index': best_c_index,
            'final_train_loss': train_losses[-1],
            'num_epochs_trained': len(train_losses),
            'feature_importances': feature_importances.tolist(),
            'sparsity_metrics': sparsity_metrics,
            'training_curves': {
                'train_loss': train_losses,
                'val_c_index': val_c_indices,
                'gradient_norms': gradient_norms
            },
            'gradient_statistics': {
                'mean': np.mean(gradient_norms),
                'std': np.std(gradient_norms),
                'max': np.max(gradient_norms),
                'min': np.min(gradient_norms)
            }
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Results for lambda={lambda_val:.5f}, l1_ratio={l1_ratio:.2f}:")
        logger.info(f"  Final C-index: {final_c_index:.4f}")
        logger.info(f"  Sparsity (< 1e-4): {sparsity_metrics['sparsity_1e-4']:.2f}%")
        logger.info(f"  Sparsity (< 1e-3): {sparsity_metrics['sparsity_1e-3']:.2f}%")
        logger.info(f"  Non-zero genes: {sparsity_metrics['num_nonzero']}")
        logger.info(f"  Importance ratio: {sparsity_metrics['importance_ratio']:.2f}x")
        logger.info(f"  Mean gradient norm: {results['gradient_statistics']['mean']:.4f}")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def _extract_feature_importance(self, model: ElasticDeepSurv) -> np.ndarray:
        """
        Extract feature importance using model's built-in method.
        """
        # Use the model's get_feature_importance method
        importance_list = model.get_feature_importance()
        
        # Convert to numpy array (sorted by feature index)
        importances = np.array([imp for _, imp in importance_list])
        
        return importances
    
    def _calculate_sparsity_metrics(self, importances: np.ndarray) -> dict:
        """
        Calculate comprehensive sparsity metrics.
        
        Returns:
            dict with multiple sparsity measures
        """
        # Different thresholds for "effectively zero"
        threshold_1e4 = 1e-4
        threshold_1e3 = 1e-3
        threshold_1e2 = 1e-2
        
        sparse_1e4 = (importances < threshold_1e4).sum()
        sparse_1e3 = (importances < threshold_1e3).sum()
        sparse_1e2 = (importances < threshold_1e2).sum()
        
        total_genes = len(importances)
        
        # Importance ratio (max / median) - should be high for sparse solutions
        importance_ratio = importances.max() / (np.median(importances) + 1e-10)
        
        # Top-k statistics
        top_10_sum = np.sort(importances)[-10:].sum()
        top_50_sum = np.sort(importances)[-50:].sum()
        total_sum = importances.sum()
        
        metrics = {
            'sparsity_1e-4': 100 * sparse_1e4 / total_genes,
            'sparsity_1e-3': 100 * sparse_1e3 / total_genes,
            'sparsity_1e-2': 100 * sparse_1e2 / total_genes,
            'num_nonzero': total_genes - sparse_1e4,
            'importance_ratio': importance_ratio,
            'top_10_concentration': 100 * top_10_sum / total_sum,
            'top_50_concentration': 100 * top_50_sum / total_sum,
            'mean_importance': importances.mean(),
            'median_importance': np.median(importances),
            'max_importance': importances.max(),
            'min_importance': importances.min(),
            'std_importance': importances.std()
        }
        
        return metrics
    
    def run_investigation(self):
        """Run complete alpha investigation."""
        logger.info("="*60)
        logger.info("PHASE 1: ALPHA PARAMETER INVESTIGATION")
        logger.info("="*60)
        
        # Load data
        self.load_data()
        
        # Store all results
        all_results = []
        
        for l1_ratio in self.l1_ratio_values:
            for lambda_val in self.lambda_values:
                try:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Training with lambda={lambda_val:.5f}, l1_ratio={l1_ratio}")
                    logger.info(f"{'='*60}")
                    
                    # Train with this (lambda, l1_ratio) combination
                    results = self.train_with_fixed_params(lambda_val, l1_ratio)
                    all_results.append(results)
                    
                    # Save individual result
                    result_file = self.output_dir / f"lambda_{lambda_val:.5f}_l1ratio_{l1_ratio:.2f}_results.json"
                    with open(result_file, 'w') as f:
                        results_serializable = self._make_json_serializable(results)
                        json.dump(results_serializable, f, indent=2)
                    
                except Exception as e:
                    logger.error(f"Error with lambda={lambda_val}, l1_ratio={l1_ratio}: {str(e)}")
                    continue
        
        # Save summary
        self._save_summary(all_results)
        
        logger.info("="*60)
        logger.info("Investigation complete!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60)
        
        return all_results
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def _save_summary(self, all_results: list):
        """Create summary table of all results."""
        if not all_results:  # ADD THIS CHECK
            logger.warning("No results to summarize")
            return
    
        summary_data = []
        
        for result in all_results:
            summary_data.append({
                'lambda': result['lambda'],        # UPDATED
                'l1_ratio': result['l1_ratio'],    # UPDATED
                'final_c_index': result['final_val_c_index'],
                'best_c_index': result['best_val_c_index'],
                'sparsity_1e-4': result['sparsity_metrics']['sparsity_1e-4'],
                'sparsity_1e-3': result['sparsity_metrics']['sparsity_1e-3'],
                'num_nonzero': result['sparsity_metrics']['num_nonzero'],
                'importance_ratio': result['sparsity_metrics']['importance_ratio'],
                'top_10_concentration': result['sparsity_metrics']['top_10_concentration'],
                'mean_grad_norm': result['gradient_statistics']['mean'],
                'max_grad_norm': result['gradient_statistics']['max'],
                'num_epochs': result['num_epochs_trained']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(['l1_ratio', 'lambda'])  # UPDATED
        
        # Save as CSV
        summary_file = self.output_dir / "hyperparameter_investigation_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info("\n" + "="*60)
        logger.info("INVESTIGATION SUMMARY")
        logger.info("="*60)
        logger.info("\n" + summary_df.to_string(index=False))
        logger.info("\n" + "="*60)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 1: Alpha Investigation')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/experiments/alpha_investigation.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: results/alpha_investigation_YYYYMMDD_HHMMSS)'
    )
    parser.add_argument(
    '--cohort',
    type=str,
    default='tcga',
    choices=['tcga', 'orien'],
    help='Cohort for hyperparameter search'
    )
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/alpha_investigation_{timestamp}"
    
    # Run investigation
    investigator = AlphaInvestigator(args.config, args.output_dir, cohort=args.cohort)
    results = investigator.run_investigation()
    
    logger.info(f"\n✅ Phase 1 complete! Results saved to: {args.output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Run: python scripts/analyze_alpha_results.py --results_dir {}".format(args.output_dir))
    logger.info("2. Review visualizations and summary report")
    logger.info("3. Decide on best alpha value for Phase 2")


if __name__ == "__main__":
    main()
