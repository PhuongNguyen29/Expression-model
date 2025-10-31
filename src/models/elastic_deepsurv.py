import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import logging

from src.models.deepsurv import DeepSurv, CoxPHLoss
from src.utils.regularization import elastic_net_penalty, get_feature_importance, count_zero_weights

logger = logging.getLogger(__name__)


class ElasticDeepSurv(DeepSurv):
    """
    DeepSurv model with Elastic Net regularization.
    """

    def __init__(self,
                 n_features: int,
                 hidden_sizes: List[int] = [256, 128],
                 dropout: float = 0.3,
                 activation: str = 'relu',
                 bacth_norm: bool = True,
                 weight_init: str = 'kaiming_uniform',   
                 l1_ratio: float = 0.5,
                 alpha: float = 0.01):
        super(ElasticDeepSurv, self).__init__(n_features, hidden_sizes, dropout)
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        assert 0.0 <= l1_ratio <= 1.0, "l1_ratio must be between 0 and 1"
        assert alpha >= 0.0, "alpha must be non-negative"
        logger.info(f"ElasticDeepSurv initialized with elastic net regularization:")
        logger.info(f"  l1_ratio: {l1_ratio} (0=Ridge, 1=Lasso)")
        logger.info(f"  alpha: {alpha}")
        logger.info(f"  Expected sparsity: {'High' if l1_ratio > 0.7 else 'Moderate' if l1_ratio > 0.3 else 'Low'}")
    

    def compute_loss(
        self,
        log_hazards: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute total loss = Cox loss + Elastic Net penalty.
        
        Total Loss = -log(Partial Likelihood) + alpha * [l1_ratio * L1 + (1-l1_ratio) * L2]
        
        Args:
            log_hazards: Model predictions (batch_size, 1)
            times: Survival times (batch_size,)
            events: Event indicators (batch_size,)
            return_components: If True, return (total_loss, cox_loss, penalty)
            
        Returns:
            total_loss: Combined loss for backpropagation
        """
        cox_criterion = CoxPHLoss()
        cox_loss = cox_criterion(log_hazards, times, events)

        # Compute Elastic Net penalty
        penalty = elastic_net_penalty(model = self, 
                                      l1_ratio = self.l1_ratio, 
                                      alpha = self.alpha,
                                      exclude_bias = True)

        # Combine losses
        total_loss = cox_loss + penalty
        
        if return_components:
            return total_loss, cox_loss, penalty
        return total_loss
    
    
    def get_sparsity_info(self) -> Dict[str, float]:
        """
        Get sparsity information of the model weights.
        
        Returns:
            Dictionary with total weights, zero weights, and sparsity percentage.
        """
        n_zeros, n_total, sparsity_ratio = count_zero_weights(self, threshold=1e-6)
        return {
            'n_zeros': n_zeros,
            'n_total': n_total,
            'sparsity_ratio': sparsity_ratio
        }

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Get feature importance scores based on model weights.
        
        Returns:
            Dictionary mapping feature index to importance score.
        """
        return get_feature_importance(self, feature_names)
        
    
class ElasticDeepSurvTrainer:
    """
    Trainer for ElasticDeepSurv model.
    Handles training loop, validation, and logging. Track Cox loss and regularization penalty separately.
    """
    
    def __init__(
        self,
        model: ElasticDeepSurv,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        scheduler_patience: int = 10,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
            )
        try:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', 
                patience=scheduler_patience, 
                factor=0.5,
                verbose=True)
            
        except TypeError:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', 
                patience=scheduler_patience, 
                factor=0.5)
            
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'train_cindex': [],
            'valid_c_index': [],
            'cox_loss': [],
            'penalty': [],
            'sparsity': []
        }
        
        logger.info("ElasticDeepSurvTrainer initialized on {device}.")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  L1 ratio: {model.l1_ratio}")
        logger.info(f"  Alpha: {model.alpha}")
        
    def train_epoch(self, train_loader) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Returns:
            avg_total_loss: Average total loss
            avg_cox_loss: Average Cox loss
            avg_penalty: Average regularization penalty
        """
        self.model.train()
        total_loss_sum = 0.0
        cox_loss_sum = 0.0
        penalty_sum = 0.0
        n_batches = 0
        
        for batch in train_loader:
            features = batch['features'].to(self.device)
            times = batch['time'].to(self.device)
            events = batch['event'].to(self.device)
            
            self.optimizer.zero_grad()
            log_hazards = self.model(features)
            total_loss, cox_loss, penalty = self.model.compute_loss(
                log_hazards, times, events, return_components=True)
            total_loss.backward()
            total_norm = 0.0
            
            for p in self.model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        raise RuntimeError("NaN detected in gradients. Check model and data for issues.")
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (0.5)
            if total_norm > 100.0:
                logger.error(f"Exploding gradients detected (norm={total_norm:.2f}). Consider reducing learning rate or adding gradient clipping.")
                return float('inf'), float('inf'), float('inf')
                
            if total_norm > 10.0:
                logger.warning(f"Large gradients detected (norm={total_norm:.2f}).")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss_sum += total_loss.item()
                cox_loss_sum += cox_loss.item()
                penalty_sum += penalty.item()
                n_batches += 1
            
        return (total_loss_sum/n_batches, cox_loss_sum/n_batches, penalty_sum/n_batches)
    
    def evaluate(self, data_loader) -> Tuple[float, float, float, float]:
        """
        Validate the model.
        
        Returns:
            avg_total_loss: Average total loss
            c_index: Concordance index
        """
        self.model.eval()
        total_loss_sum = 0.0
        cox_loss_sum = 0.0
        penalty_sum = 0.0
        n_batches = 0
        
        all_risks = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                times = batch['time'].to(self.device)
                events = batch['event'].to(self.device)
                
                log_hazards = self.model(features)
                total_loss, cox_loss, penalty = self.model.compute_loss(log_hazards, times, events, return_components=True)
                
                total_loss_sum += total_loss.item()
                cox_loss_sum += cox_loss.item()
                penalty_sum += penalty.item()   
                n_batches += 1
                risks = torch.exp(log_hazards).squeeze().cpu().numpy()
                all_risks.extend(risks)
                all_times.extend(times.cpu().numpy())
                all_events.extend(events.cpu().numpy())
                
        from lifelines.utils import concordance_index 
        try:
            c_index = concordance_index(all_times, -np.array(all_risks), all_events)
        except Exception as e:
            logger.error(f"Error computing concordance index: {e}")
            raise RuntimeError("Failed to compute concordance index during validation.")
        

        return (total_loss_sum / n_batches, 
                cox_loss_sum / n_batches, 
                penalty_sum / n_batches ,
                c_index)
        
    def fit(
        self,
        train_loader,
        valid_loader,
        n_epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose:bool = True
    ):
        """
         Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            n_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        best_cindex = 0.0
        patience_counter = 0
        best_model_state = None
        
        logger.info("Starting training...")
        logger.info(f"Max epochs: {n_epochs}, Early stopping patience: {early_stopping_patience}")
        
        for epoch in range(n_epochs):
            train_total, train_cox, train_penalty = self.train_epoch(train_loader)
            valid_total, valid_cox, valid_penalty, valid_cindex = self.evaluate(valid_loader)
            
            _, _, _, train_cindex = self.evaluate(train_loader)
            
            sparsity_info = self.model.get_sparsity_info()
            sparsity_ratio = sparsity_info['sparsity_ratio']
            
            #store history
            self.history['train_loss'].append(train_total)
            self.history['valid_loss'].append(valid_total)
            self.history['train_cindex'].append(train_cindex)
            self.history['valid_c_index'].append(valid_cindex)
            self.history['cox_loss'].append(train_cox)
            self.history['penalty'].append(train_penalty)
            self.history['sparsity'].append(sparsity_ratio)
            
            self.scheduler.step(valid_cindex)
            
            #early stoping
            if valid_cindex > best_cindex:
                best_cindex = valid_cindex
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            #Print progress
            if verbose:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Train Loss: {train_total:.4f} (Cox: {train_cox:.4f}, Penalty: {train_penalty:.4f}) | "
                    f"C-index: {train_cindex:.4f} | "
                    f"Valid C-index: {valid_cindex:.4f} | "
                    f"Sparsity: {sparsity_ratio:.1%}"
                )
        
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break
            
        #Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Best model with C-index: {best_cindex:.4f} restored.")
            
            final_sparsity = self.model.get_sparsity_info()
            logger.info(f"Final model sparsity: {final_sparsity['sparsity_ratio']:.1%} "
                        f"({final_sparsity['n_zeros']} / {final_sparsity['n_total']} weights zeroed).")
        return self.history
    
    
    #Example usgae and testing
if __name__ == "__main__":
    print("="*60)
    print("Testing ElasticDeepSurv model and trainer...")
    print("="*60 )
    model = ElasticDeepSurv(n_features=100, hidden_sizes=[64,32], dropout=0.3, l1_ratio=0.7, alpha=0.01)
    
    print(f"n\Model architecture:\n{model}")
    print("n\Testing forward pass...")
    x = torch.randn(16, 100)
    output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    print("n\Testing loss computation...")
    times = torch.rand(16)*100
    events = torch.randint(0,2,(16,)).float()
    total_loss, cox_loss, penalty = model.compute_loss(output, times, events, return_components=True)
    
    print(f"Total loss: {total_loss.item():.4f}, Cox loss: {cox_loss.item():.4f}, Penalty: {penalty.item():.4f}")
    
    #test sparsity
    print("\nInitial sparsity:")
    sparsity = model.get_sparsity_info()
    print(f"  Near-zero weights: {sparsity['n_zeros']}/{sparsity['n_total']}")
    print(f"  Sparsity ratio: {sparsity['sparsity_ratio']:.1%}")
     
    print("\nTop 5 feature importances:")
    gene_names = [f"gene_{i}" for i in range(100)]
    importances = model.get_feature_importance(gene_names)
    for gene, score in importances[:5]:
        print(f"  {gene}: {score:.4f}")
        
    print("\n" + "="*60)
    print("All tests passed for ElasticDeepSurv.")
    print("="*60)