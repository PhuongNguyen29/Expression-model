"""
Proximal Operators for Sparse Optimization

Based on:
- Beck & Teboulle (2009). "A fast iterative shrinkage-thresholding algorithm 
  for linear inverse problems." SIAM J Imaging Sciences.
- Parikh & Boyd (2014). "Proximal algorithms." Found Trends Optim.
- Yuan & Lin (2006). "Model selection and estimation in regression with 
  grouped variables."
"""

import torch
import torch.nn as nn
import numpy as np


def soft_threshold(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """
    Soft-thresholding operator for Lasso (element-wise sparsity).
    
    prox(x) = sign(x) * max(|x| - λ, 0)
    
    Args:
        x: Input tensor
        lambda_: Threshold parameter
        
    Returns:
        Soft-thresholded tensor
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - lambda_, min=0.0)


def group_soft_threshold(x: torch.Tensor, lambda_: float, dim: int = 0) -> torch.Tensor:
    """
    Group soft-thresholding operator for Group Lasso.
    
    For a group of weights w, the proximal operator is:
        prox(w) = (1 - λ/||w||_2) * w  if ||w||_2 > λ
        prox(w) = 0                     if ||w||_2 ≤ λ
    
    This sets entire groups to zero when their L2 norm is below threshold.
    
    Based on: Yuan & Lin (2006), Equation 3.4
    
    Args:
        x: Weight matrix [output_dim, input_dim]
        lambda_: Regularization strength
        dim: Dimension along which to group (0 = group by columns/genes)
        
    Returns:
        Thresholded weights with entire groups set to zero
        
    Example:
        W shape: [256 hidden, 308 genes]
        dim=0 groups by input features (genes)
        Each column can be entirely zeroed out
    """
    # Compute L2 norm for each group
    group_norms = torch.norm(x, p=2, dim=dim, keepdim=True)  # [1, 308] if dim=0
    
    # Compute shrinkage factor
    shrinkage = torch.clamp(1.0 - lambda_ / (group_norms + 1e-10), min=0.0)
    
    # Apply group-wise shrinkage
    return x * shrinkage


def sparse_group_soft_threshold(
    x: torch.Tensor,
    lambda_group: float,
    lambda_sparse: float,
    dim: int = 0
) -> torch.Tensor:
    """
    Sparse Group Lasso proximal operator.
    Combines group sparsity + element-wise sparsity.
    
    Based on: Simon et al. (2013) "A sparse-group lasso"
    
    Args:
        x: Weight matrix
        lambda_group: Group Lasso parameter
        lambda_sparse: Lasso parameter (element-wise)
        dim: Grouping dimension
        
    Returns:
        Weights with both group and element-wise sparsity
    """
    # Step 1: Element-wise soft thresholding
    x_sparse = soft_threshold(x, lambda_sparse)
    
    # Step 2: Group soft thresholding
    x_group_sparse = group_soft_threshold(x_sparse, lambda_group, dim=dim)
    
    return x_group_sparse


class ProximalOperator:
    """
    Wrapper class for proximal operators to use with PyTorch optimizers.
    """
    
    def __init__(self, operator_type: str = 'group_lasso', lambda_: float = 0.01):
        """
        Args:
            operator_type: 'lasso', 'group_lasso', or 'sparse_group_lasso'
            lambda_: Regularization strength
        """
        self.operator_type = operator_type
        self.lambda_ = lambda_
        
    def apply(self, param: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Apply proximal operator to parameter."""
        if self.operator_type == 'lasso':
            return soft_threshold(param, self.lambda_)
        elif self.operator_type == 'group_lasso':
            return group_soft_threshold(param, self.lambda_, dim=dim)
        elif self.operator_type == 'sparse_group_lasso':
            return sparse_group_soft_threshold(
                param, 
                lambda_group=self.lambda_,
                lambda_sparse=self.lambda_ * 0.5,  # Mix: 50% group, 50% element
                dim=dim
            )
        else:
            raise ValueError(f"Unknown operator type: {self.operator_type}")


def apply_proximal_operator_to_model(
    model: nn.Module,
    operator: ProximalOperator,
    first_layer_only: bool = True
) -> None:
    """
    Apply proximal operator to model parameters IN-PLACE.
    
    This is called after each gradient step:
        1. θ_temp = θ - lr × ∇loss  (gradient descent)
        2. θ_new = prox(θ_temp)      (proximal operator)
    
    Args:
        model: Neural network
        operator: Proximal operator to apply
        first_layer_only: Only apply to first layer (input → hidden)
    """
    layer_count = 0
    
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
            
        if first_layer_only and layer_count > 0:
            break
            
        # Apply proximal operator
        # For Group Lasso, group by input features (columns, dim=0)
        with torch.no_grad():
            param.data = operator.apply(param.data, dim=0)
            
        layer_count += 1


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing Proximal Operators")
    print("="*60)
    
    # Test soft thresholding
    x = torch.randn(5)
    print(f"\nOriginal: {x}")
    print(f"Soft threshold (λ=0.5): {soft_threshold(x, 0.5)}")
    
    # Test group soft thresholding
    W = torch.randn(4, 3)  # 4 hidden units, 3 input features
    print(f"\nOriginal weights:\n{W}")
    print(f"\nGroup soft threshold (λ=1.0):\n{group_soft_threshold(W, 1.0, dim=0)}")
    
    # Check which groups went to zero
    result = group_soft_threshold(W, 1.0, dim=0)
    zero_groups = (result.abs().sum(dim=0) < 1e-6).sum().item()
    print(f"\nNumber of zero groups: {zero_groups}/3")
    
    print("\n" + "="*60)