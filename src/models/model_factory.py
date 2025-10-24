"""
Model Factory for Flexible Model Creation
Supports multiple survival models: DeepSurv, Cox-PASNet, CoxNet, etc.
Based on Factory Design Pattern (Gang of Four, 1994)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

# Import your existing models
from src.models.deepsurv import DeepSurv

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating different survival models.
    
    Supported models:
    - deepsurv: Feed-forward neural network (Katzman et al., 2018)
    - coxnet: Elastic net Cox model (future)
    - cox_pasnet: Pathway-guided model (future)
    - deephit: Competing risks model (future)
    """
    
    SUPPORTED_MODELS = ['deepsurv', 'coxnet', 'cox_pasnet', 'deephit']
    
    @staticmethod
    def create_model(model_type: str, n_features: int, config: Dict) -> nn.Module:
        """
        Create a survival model based on configuration.
        
        Args:
            model_type: Type of model ('deepsurv', 'coxnet', etc.)
            n_features: Number of input features
            config: Model configuration dict
            
        Returns:
            Initialized model
        """
        if model_type not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {ModelFactory.SUPPORTED_MODELS}"
            )
        
        logger.info(f"Creating model: {model_type}")
        
        if model_type == 'deepsurv':
            return ModelFactory._create_deepsurv(n_features, config)
        elif model_type == 'coxnet':
            raise NotImplementedError("CoxNet not yet implemented")
        elif model_type == 'cox_pasnet':
            raise NotImplementedError("Cox-PASNet not yet implemented")
        elif model_type == 'deephit':
            raise NotImplementedError("DeepHit not yet implemented")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _create_deepsurv(n_features: int, config: Dict) -> DeepSurv:
        """
        Create DeepSurv model.
        
        Based on: Katzman et al., 2018, BMC Medical Research Methodology
        """
        model_config = config.get('model', {})
        
        model = DeepSurv(
            n_features=n_features,
            hidden_sizes=model_config.get('hidden_sizes', [512, 256]),
            dropout=model_config.get('dropout', 0.4),
            activation=model_config.get('activation', 'relu'),
            batch_norm=model_config.get('batch_norm', True),
            weight_init=model_config.get('weight_init', 'xavier_uniform')
        )
        
        logger.info(f"  DeepSurv created with {n_features} input features")
        logger.info(f"  Architecture: {model_config.get('hidden_sizes', [512, 256])}")
        logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    # Future model creation methods
    @staticmethod
    def _create_coxnet(n_features: int, config: Dict):
        """Create CoxNet model (elastic net regularized Cox)."""
        # TODO: Implement when needed
        pass
    
    @staticmethod
    def _create_cox_pasnet(n_features: int, config: Dict, pathway_mask: Optional[torch.Tensor] = None):
        """
        Create Cox-PASNet model (pathway-guided).
        
        Based on: Hao et al., 2018, Bioinformatics
        """
        # TODO: Implement when needed
        pass


class TrainerFactory:
    """
    Factory for creating model trainers with consistent interface.
    """
    
    @staticmethod
    def create_trainer(model: nn.Module, config: Dict, device: str = 'cuda'):
        """
        Create trainer for the model.
        
        Args:
            model: PyTorch model
            config: Training configuration dict
            device: Device to train on
            
        Returns:
            Trainer instance
        """
        model_type = config['model']['type']
        
        if model_type == 'deepsurv':
            from src.models.deepsurv import DeepSurvTrainer
            
            training_config = config.get('training', {})
            
            trainer = DeepSurvTrainer(
                model=model,
                learning_rate=training_config.get('learning_rate', 0.001),
                weight_decay=training_config.get('weight_decay', 0.01),
                device=device
            )
            
            logger.info(f"  Trainer created for {model_type}")
            logger.info(f"  Learning rate: {training_config.get('learning_rate', 0.001)}")
            logger.info(f"  Weight decay: {training_config.get('weight_decay', 0.01)}")
            
            return trainer
        else:
            raise NotImplementedError(f"Trainer for {model_type} not yet implemented")


def create_model_from_config(config: Dict, n_features: int) -> tuple:
    """
    Convenience function to create model and trainer from config.
    
    Args:
        config: Experiment configuration dict
        n_features: Number of input features
        
    Returns:
        (model, trainer) tuple
    """
    # Get device
    device_config = config.get('compute', {}).get('device', 'auto')
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = device_config
    
    logger.info(f"Using device: {device}")
    
    # Create model
    model_type = config['model']['type']
    model = ModelFactory.create_model(model_type, n_features, config)
    
    # Create trainer
    trainer = TrainerFactory.create_trainer(model, config, device)
    
    return model, trainer


# Example usage
if __name__ == "__main__":
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open("config/experiments/deepsurv_full.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    n_features = 14778  # Example
    model, trainer = create_model_from_config(config, n_features)
    
    print(f"\nModel type: {config['model']['type']}")
    print(f"Model: {model}")
