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
from sklearn.model_selection import StratifiedKFold

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_factory import load_dataset_from_config
from src.models.elastic_deepsurv import ElasticDeepSurv
from src.utils.batch_samplers import StratifiedSurvivalSampler
from lifelines.utils import concordance_index

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlphaInvestigator:
    """
    Systematic investigation of alpha parameter effects.
    
    Tests fixed alpha values to understand:
    1. Alpha-sparsity relationship
    2. Alpha-performance trade-off
    3. Gradient stability across alpha values
    """
    
    def __init__(self, config_path: str, output_dir: str):
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
        self.alpha_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        
        logger.info(f"Testing {len(self.alpha_values)} alpha values: {self.alpha_values}")
        
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
        
        # Use TCGA only for Phase 1 (faster iteration)
        self.tcga_expr = data['tcga_expr']
        self.surv_tcga = data['surv_tcga']
        
        # Convert to tensors
        self.X = torch.FloatTensor(self.tcga_expr.T.values).to(self.device)
        self.T = torch.FloatTensor(self.surv_tcga['time'].values).to(self.device)
        self.E = torch.FloatTensor(self.surv_tcga['event'].values).to(self.device)
        
        self.n_features = self.X.shape[1]
        
        logger.info(f"TCGA samples: {self.X.shape[0]}")
        logger.info(f"Features (genes): {self.n_features}")
        logger.info(f"Events: {self.E.sum().item()}/{len(self.E)} ({100*self.E.mean().item():.1f}%)")
        logger.info("="*60)
    
    def train_with_fixed_alpha(self, alpha: float, fold: int = 0) -> dict:
        """
        Train model with fixed alpha value.
        
        Args:
            alpha: Fixed alpha value to test
            fold: Cross-validation fold number
            
        Returns:
            dict with training results and metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with alpha={alpha:.4f}")
        logger.info(f"{'='*60}")
        
        # Fixed hyperparameters from your current best configuration
        config = {
            'input_dim': self.n_features,
            'hidden_layers': [256, 64],  # Your TCGA best architecture
            'dropout': 0.3,
            'activation': 'relu',
            'batch_norm': True,
            'alpha': alpha,
            'l1_ratio': 0.7,  # Your current best
        }
        
        # Training parameters
        learning_rate = 0.000337  # Your TCGA best
        num_epochs = 100  # Reduced from 500 for faster iteration
        batch_size = 32
        patience = 20  # Early stopping
        
        # Create model
        model = ElasticDeepSurv(**config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Split data (80/20 train/val)
        n_samples = len(self.X)
        n_train = int(0.8 * n_samples)
        
        # Stratified split by event status
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        X_train, X_val = self.X[train_idx], self.X[val_idx]
        T_train, T_val = self.T[train_idx], self.T[val_idx]
        E_train, E_val = self.E[train_idx], self.E[val_idx]
        
        # Training metrics storage
        train_losses = []
        val_c_indices = []
        gradient_norms = []
        
        best_c_index = 0
        patience_counter = 0
        best_state = None
        
        logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        for epoch in range(num_epochs):
            model.train()
            
            # Create batches
            n_batches = (len(X_train) + batch_size - 1) // batch_size
            epoch_loss = 0
            epoch_grad_norm = 0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X_train))
                
                X_batch = X_train[start_idx:end_idx]
                T_batch = T_train[start_idx:end_idx]
                E_batch = E_train[start_idx:end_idx]
                
                # Forward pass
                risk_scores = model(X_batch)
                
                # Cox loss
                loss = model.cox_loss(risk_scores, T_batch, E_batch)
                
                # Add elastic net penalty (already in model)
                loss = loss  # Model handles regularization internally
                
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
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / n_batches
            avg_grad_norm = epoch_grad_norm / n_batches
            train_losses.append(avg_loss)
            gradient_norms.append(avg_grad_norm)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_risks = model(X_val)
                val_c_index = concordance_index(
                    T_val.cpu().numpy(),
                    -val_risks.cpu().numpy().flatten(),
                    E_val.cpu().numpy()
                )
                val_c_indices.append(val_c_index)
            
            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
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
            'alpha': alpha,
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
        logger.info(f"Results for alpha={alpha:.4f}:")
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
        Extract feature importance from first layer weights.
        Method: L1 norm of weights for each input feature.
        
        Based on: Lundberg & Lee (2017) "A unified approach to interpreting model predictions"
        """
        first_layer_weights = model.fc1.weight.data.cpu().numpy()  # Shape: [hidden_dim, input_dim]
        
        # L1 norm across hidden units for each input feature
        feature_importance = np.abs(first_layer_weights).sum(axis=0)  # Shape: [input_dim]
        
        return feature_importance
    
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
        
        # Test each alpha value
        for alpha in self.alpha_values:
            try:
                results = self.train_with_fixed_alpha(alpha)
                all_results.append(results)
                
                # Save individual result
                result_file = self.output_dir / f"alpha_{alpha:.4f}_results.json"
                with open(result_file, 'w') as f:
                    # Convert numpy types to Python types for JSON
                    results_serializable = self._make_json_serializable(results)
                    json.dump(results_serializable, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error with alpha={alpha}: {str(e)}")
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
        summary_data = []
        
        for result in all_results:
            summary_data.append({
                'alpha': result['alpha'],
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
        summary_df = summary_df.sort_values('alpha')
        
        # Save as CSV
        summary_file = self.output_dir / "alpha_investigation_summary.csv"
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
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/alpha_investigation_{timestamp}"
    
    # Run investigation
    investigator = AlphaInvestigator(args.config, args.output_dir)
    results = investigator.run_investigation()
    
    logger.info(f"\n✅ Phase 1 complete! Results saved to: {args.output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Run: python scripts/analyze_alpha_results.py --results_dir {}".format(args.output_dir))
    logger.info("2. Review visualizations and summary report")
    logger.info("3. Decide on best alpha value for Phase 2")


if __name__ == "__main__":
    main()
