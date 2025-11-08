import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.elastic_deepsurv import ElasticDeepSurv

# Create test model
model = ElasticDeepSurv(
    n_features=308,
    hidden_sizes=[256, 64],
    dropout=0.3,
    activation='relu',
    batch_norm=True,
    alpha=0.01,
    l1_ratio=0.7
)

print("Model Parameter Structure:")
print("="*60)

weight_matrix_count = 0
for name, param in model.named_parameters():
    param_type = "Weight matrix" if param.dim() >= 2 else "Bias/BatchNorm"
    
    if param.dim() >= 2:
        layer_marker = " ← FIRST LAYER (PROXIMAL APPLIED)" if weight_matrix_count == 0 else ""
        print(f"{name:30s} | Shape: {str(list(param.shape)):20s} | {param_type}{layer_marker}")
        weight_matrix_count += 1
    else:
        print(f"{name:30s} | Shape: {str(list(param.shape)):20s} | {param_type}")

print("="*60)
print(f"\nTotal weight matrices: {weight_matrix_count}")
print("First weight matrix will receive Group Lasso proximal operator")