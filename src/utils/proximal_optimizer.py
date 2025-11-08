"""
Proximal Gradient Descent Optimizer

Implements: Proximal Gradient Descent (also called ISTA)
Based on: Beck & Teboulle (2009) FISTA algorithm
"""

import torch
from torch.optim import Optimizer
from typing import Callable, Optional
from .proximal_operators import ProximalOperator
import numpy as np


class ProximalGradientDescent(Optimizer):
    """
    Proximal Gradient Descent optimizer for sparse neural networks.
    
    Algorithm:
        1. Gradient step: θ_temp = θ - lr × ∇loss
        2. Proximal step: θ_new = prox(θ_temp, λ)
    
    The proximal step induces sparsity (exact zeros).
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        prox_operator: Optional[ProximalOperator] = None,
        momentum: float = 0.0
    ):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            prox_operator: Proximal operator (e.g., Group Lasso)
            momentum: Momentum coefficient (0 = no momentum)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
            
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        
        self.prox_operator = prox_operator
        
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        
        CRITICAL: Proximal operator applied ONLY to first weight matrix (input layer)
        for gene-level sparsity. Based on Feng et al. (2019).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_matrix_count = 0  # Track weight matrices ONLY
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize momentum buffer
                if len(state) == 0 and group['momentum'] > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                # Gradient descent step with momentum
                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(grad)
                    p.data.add_(buf, alpha=-group['lr'])
                else:
                    p.data.add_(grad, alpha=-group['lr'])
                
                # Proximal operator step - ONLY on first weight matrix
                # Applies to input→hidden layer for gene-level sparsity
                if self.prox_operator is not None and p.dim() >= 2:
                    if weight_matrix_count == 0:  # First layer only
                        p.data = self.prox_operator.apply(p.data, dim=0)
                    weight_matrix_count += 1
        
        return loss


class AcceleratedProximalGradient(Optimizer):
    """
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
    
    Accelerated version of proximal gradient descent.
    Converges faster: O(1/k²) vs O(1/k) for standard proximal gradient.
    
    Based on: Beck & Teboulle (2009)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        prox_operator: Optional[ProximalOperator] = None
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        
        self.prox_operator = prox_operator
        self.k = 0  # Iteration counter
        
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs FISTA step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.k += 1
        t_new = (1 + np.sqrt(1 + 4 * ((self.k - 1) ** 2))) / 2
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # Initialize
                if len(state) == 0:
                    state['y'] = p.data.clone()
                    state['x_old'] = p.data.clone()
                    state['t_old'] = 1.0
                
                y = state['y']
                x_old = state['x_old']
                t_old = state['t_old']
                
                # Gradient step at momentum point
                y.add_(p.grad.data, alpha=-group['lr'])
                
                # Proximal step
                if self.prox_operator is not None and p.dim() >= 2:
                    x_new = self.prox_operator.apply(y.clone(), dim=0)
                else:
                    x_new = y.clone()
                
                # Acceleration (Nesterov momentum)
                momentum = (t_old - 1) / t_new
                y.copy_(x_new + momentum * (x_new - x_old))
                
                # Update
                p.data.copy_(x_new)
                state['x_old'].copy_(x_new)
                state['t_old'] = t_new
        
        return loss


# Helper function
def create_proximal_optimizer(
    model_params,
    optimizer_type: str = 'proximal_gd',
    lr: float = 1e-3,
    alpha: float = 0.01,
    l1_ratio: float = 0.7,
    use_group_lasso: bool = True,
    lambda_scale: float = None 
) -> Optimizer:
    """
    Factory function to create proximal optimizer.
    
    Args:
        model_params: Model parameters to optimize
        optimizer_type: 'proximal_gd' or 'fista'
        lr: Learning rate
        alpha: Regularization strength
        l1_ratio: Mix between group lasso and L2 (higher = more group lasso)
        use_group_lasso: Use Group Lasso vs element-wise Lasso
        
    Returns:
        Configured optimizer with proximal operator
    """
    # Create proximal operator
    operator_type = 'group_lasso' if use_group_lasso else 'lasso'
    if lambda_scale is not None:
        effective_lambda = lambda_scale
    else:
        effective_lambda = alpha * l1_ratio * lr * 0.01  # Default 100x smaller

    prox_op = ProximalOperator(
        operator_type=operator_type,
        lambda_=effective_lambda
    )
    
    # Create optimizer
    if optimizer_type == 'proximal_gd':
        optimizer = ProximalGradientDescent(
            model_params,
            lr=lr,
            prox_operator=prox_op,
            momentum=0.9  # Can add momentum
        )
    elif optimizer_type == 'fista':
        optimizer = AcceleratedProximalGradient(
            model_params,
            lr=lr,
            prox_operator=prox_op
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer