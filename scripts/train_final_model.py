"""
Train Final ElasticDeepSurv Model on Full Cohort
================================================

Trains model on entire cohort (no train/val split) using optimal hyperparameters
identified from grid search. Saves model and extracts gene importances.

Usage:
    python scripts/train_final_model.py --cohort tcga --lambda 0.0005 --l1_ratio 0.7
    python scripts/train_final_model.py --cohort orien --lambda 0.0005 --l1_ratio 0.7

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


# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_factory import load_dataset_from_config
from src.models.elastic_deepsurv import ElasticDeepSurv
from torch.utils.data import TensorDataset, DataLoader
from src.utils.batch_samplers import StratifiedBatchSampler
from src.utils.proximal_optimizer import create_proximal_optimizer
from src.models.deepsurv import CoxPHLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinalModelTrainer:
    """
    Train final ElasticDeepSurv model on complete cohort.
    
    No train/val split - uses all available samples with optimal hyperparameters.
    """
    
    def __init__(
        self,
        config_path: str,
        cohort: str,
        lambda_val: float,
        l1_ratio: float,
        output_dir: str
    ):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to experiment config
            cohort: 'tcga' or 'orien'
            lambda_val: Optimal lambda from grid search
            l1_ratio: Optimal l1_ratio from grid search
            output_dir: Directory to save results
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cohort = cohort.lower()
        assert self.cohort in ['tcga', 'orien'], "Cohort must be 'tcga' or 'orien'"
        
        self.lambda_val = lambda_val
        self.l1_ratio = l1_ratio
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set seeds for reproducibility
        self.seed = self.config['project']['seed']
        self._set_seeds()
        
        logger.info(f"Training final model on {self.cohort.upper()}")
        logger.info(f"Hyperparameters: lambda={lambda_val}, l1_ratio={l1_ratio}")
    
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
        """Load full cohort data (no split)."""
        logger.info("="*60)
        logger.info(f"Loading {self.cohort.upper()} data...")
        
        data = load_dataset_from_config(self.config)
        
        # Select cohort
        if self.cohort == 'tcga':
            expr_data = data['tcga_expr']
            surv_data = data['surv_tcga']
        else:  # orien
            expr_data = data['orien_expr']
            surv_data = data['surv_orien']
        
        # Convert to tensors - USE ALL SAMPLES
        self.X = torch.FloatTensor(expr_data.T.values).to(self.device)
        self.T = torch.FloatTensor(surv_data['time'].values).to(self.device)
        self.E = torch.FloatTensor(surv_data['event'].values).to(self.device)
        
        self.n_features = self.X.shape[1]
        self.n_samples = self.X.shape[0]
        
        logger.info(f"{self.cohort.upper()} samples: {self.n_samples}")
        logger.info(f"Features (genes): {self.n_features}")
        logger.info(f"Events: {self.E.sum().item()}/{len(self.E)} ({100*self.E.mean().item():.1f}%)")
        logger.info("="*60)
    
    def train(self):
        """Train model on full cohort."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Final Model - {self.cohort.upper()}")
        logger.info(f"{'='*60}")
        
        # Model configuration
        model_config = {
            'n_features': self.n_features,
            'hidden_sizes': [256, 64],
            'dropout': 0.3,
            'activation': 'relu',
            'batch_norm': True,
            'alpha': self.lambda_val,
            'l1_ratio': self.l1_ratio,
        }
        
        # Training parameters
        learning_rate = 0.000337
        num_epochs = 150  # More epochs since no early stopping
        batch_size = 32
        
        # Create model
        model = ElasticDeepSurv(**model_config).to(self.device)
        
        # Create optimizer
        optimizer = create_proximal_optimizer(
            model.parameters(),
            optimizer_type='fista',
            lr=learning_rate,
            alpha=0.1,  # Dummy
            l1_ratio=self.l1_ratio,
            use_group_lasso=True,
            lambda_scale=self.lambda_val
        )
        
        # Create dataloader (no split - use all data)
        dataset = TensorDataset(self.X, self.T, self.E)
        
        sampler = StratifiedBatchSampler(
            events=self.E.cpu().numpy(),
            batch_size=batch_size,
            shuffle=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0
        )
        
        logger.info(f"Training on ALL {self.n_samples} samples (no validation split)")
        logger.info(f"Epochs: {num_epochs}, Batch size: {batch_size}")
        
        # Training metrics
        train_losses = []
        gradient_norms = []
        gene_sparsity_history = []
        
        train_losses = []
        gradient_norms = []
        gene_sparsity_history = []

        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
        best_model_state = None
        best_epoch = 0

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            epoch_grad_norm = 0
            n_batches = 0
            
            for batch_data in dataloader:
                X_batch, T_batch, E_batch = batch_data
                
                # Forward pass
                risk_scores = model(X_batch)
                
                # Cox loss only (proximal handles regularization)
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
            
            # Check gene collapse EVERY epoch
            first_layer_weight = None
            for name, param in model.named_parameters():
                if 'fc0.weight' in name:
                    first_layer_weight = param.data
                    break
            
            current_active_genes = 0
            if first_layer_weight is not None:
                with torch.no_grad():
                    gene_norms = torch.norm(first_layer_weight, p=2, dim=0)
                    zero_genes = (gene_norms < 1e-4).sum().item()
                    current_active_genes = len(gene_norms) - zero_genes
            
            # Early stopping: Only save if BOTH loss improved AND genes not collapsed
            if avg_loss < best_loss and current_active_genes >= 250:
                best_loss = avg_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"  [Epoch {epoch+1}/{num_epochs}] "
                    f"Loss: {avg_loss:.4f} | "
                    f"Grad: {avg_grad_norm:.4f} | "
                    f"Active genes: {current_active_genes}/308 | "
                    f"Patience: {patience_counter}/{patience}"
                )
            
                gene_sparsity_history.append({
                    'epoch': epoch + 1,
                    'zero_genes': zero_genes,
                    'active_genes': current_active_genes,
                    'sparsity_pct': 100 * zero_genes / len(gene_norms)
                })
            
            # Check for early stopping
            if patience_counter >= patience:
                logger.info(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                logger.info(f"Best model from epoch {best_epoch} with loss: {best_loss:.4f}")
                break

        # RESTORE BEST MODEL
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"\n✅ Restored model from epoch {best_epoch} (loss: {best_loss:.4f})")
            
            # Verify restored model has active genes
            first_layer_weight = None
            for name, param in model.named_parameters():
                if 'fc0.weight' in name:
                    first_layer_weight = param.data
                    break
            
            if first_layer_weight is not None:
                with torch.no_grad():
                    gene_norms = torch.norm(first_layer_weight, p=2, dim=0)
                    zero_genes = (gene_norms < 1e-4).sum().item()
                    active_genes = len(gene_norms) - zero_genes
                    logger.info(f"   Verified: {active_genes}/308 active genes in restored model")
                    logger.info(f"   Gene norms: min={gene_norms.min():.6f}, max={gene_norms.max():.6f}, mean={gene_norms.mean():.6f}")
        else:
            logger.warning("⚠️  No best model state saved - using final epoch model")
        
        # Save final model
        model_path = self.output_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'hyperparameters': {
                'lambda': self.lambda_val,
                'l1_ratio': self.l1_ratio,
                'learning_rate': learning_rate
            },
            'training_info': {
                'cohort': self.cohort,
                'n_samples': self.n_samples,
                'n_features': self.n_features,
                'n_epochs': len(train_losses),
                'best_epoch': best_epoch,
                'final_loss': train_losses[-1],
                'best_loss': best_loss
            }
        }, model_path)
        
        logger.info(f"Model saved to: {model_path}")
        
        # Extract and save gene importances
        self._extract_and_save_importances(model)
        
        # Save training history
        self._save_training_history(train_losses, gradient_norms, gene_sparsity_history)
        
        return model
    
    def _extract_and_save_importances(self, model):
        """Extract gene importances and save to file."""
        logger.info("\nExtracting gene importances...")
        
        model.eval()
        
        # Get first layer weights
        first_layer_weight = None
        for name, param in model.named_parameters():
            if 'fc0.weight' in name:
                first_layer_weight = param.data
                break
        
        if first_layer_weight is None:
            logger.error("Could not find first layer weights!")
            return
        
        with torch.no_grad():
            # Compute gene-level importance (L2 norm of weights)
            gene_norms = torch.norm(first_layer_weight, p=2, dim=0).cpu().numpy()
        
        # Create dataframe with gene names if available
        if self.cohort == 'tcga':
            # Load gene names from data
            data = load_dataset_from_config(self.config)
            gene_names = data['tcga_expr'].index.tolist()
        else:
            # ORIEN
            data = load_dataset_from_config(self.config)
            gene_names = data['orien_expr'].index.tolist()
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'gene': gene_names,
            'importance': gene_norms
        })
        
        # Sort by importance (descending)
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        # Save to CSV
        importance_path = self.output_dir / 'gene_importances.csv'
        importance_df.to_csv(importance_path, index=False)
        
        logger.info(f"Gene importances saved to: {importance_path}")
        logger.info(f"\nTop 10 genes by importance:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['rank']:2d}. {row['gene']:20s} {row['importance']:.6f}")
        
        # Save summary statistics
        summary = {
            'total_genes': len(gene_norms),
            'zero_genes_1e-4': int((gene_norms < 1e-4).sum()),
            'zero_genes_1e-3': int((gene_norms < 1e-3).sum()),
            'active_genes': int((gene_norms >= 1e-4).sum()),
            'importance_stats': {
                'min': float(gene_norms.min()),
                'max': float(gene_norms.max()),
                'mean': float(gene_norms.mean()),
                'median': float(np.median(gene_norms)),
                'std': float(gene_norms.std())
            },
            'top_10_genes': importance_df.head(10)['gene'].tolist()
        }
        
        summary_path = self.output_dir / 'importance_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nImportance summary:")
        logger.info(f"  Total genes: {summary['total_genes']}")
        logger.info(f"  Active genes (≥1e-4): {summary['active_genes']}")
        logger.info(f"  Zero genes (<1e-4): {summary['zero_genes_1e-4']}")
    
    def _save_training_history(self, train_losses, gradient_norms, gene_sparsity_history):
        """Save training history to file."""
        history = {
            'train_losses': train_losses,
            'gradient_norms': gradient_norms,
            'gene_sparsity': gene_sparsity_history,
            'hyperparameters': {
                'lambda': self.lambda_val,
                'l1_ratio': self.l1_ratio
            },
            'cohort': self.cohort,
            'n_samples': self.n_samples
        }
        
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved to: {history_path}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train final ElasticDeepSurv model on full cohort'
    )
    parser.add_argument(
        '--cohort',
        type=str,
        required=True,
        choices=['tcga', 'orien'],
        help='Cohort to train on: tcga or orien'
    )
    parser.add_argument(
        '--lambda',
        type=float,
        dest='lambda_val',
        required=True,
        help='Optimal lambda value from grid search (e.g., 0.0005)'
    )
    parser.add_argument(
        '--l1_ratio',
        type=float,
        required=True,
        help='Optimal l1_ratio value from grid search (e.g., 0.7)'
    )
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
        help='Output directory (default: results/final_model_{cohort}_{timestamp})'
    )
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/final_model_{args.cohort}_{timestamp}"
    
    # Train model
    trainer = FinalModelTrainer(
        config_path=args.config,
        cohort=args.cohort,
        lambda_val=args.lambda_val,
        l1_ratio=args.l1_ratio,
        output_dir=args.output_dir
    )
    
    trainer.load_data()
    model = trainer.train()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Training complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"{'='*60}")
    logger.info("\nNext steps:")
    logger.info(f"1. Check gene importances: {args.output_dir}/gene_importances.csv")
    logger.info(f"2. Train on other cohort if needed")
    logger.info(f"3. Run cross-cohort testing for bidirectional validation")


if __name__ == "__main__":
    main()