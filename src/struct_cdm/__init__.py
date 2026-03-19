"""struct-sdfm — Hierarchical TabICL with LCS synthetic data prior."""

from .model     import TabICLModel, create_model
from .predictor import HierarchicalTabICLClassifier, HierarchicalTabICLRegressor
from .train     import train, load_checkpoint
from .prior     import generate_dataset, generate_batch

__all__ = [
    'TabICLModel',
    'create_model',
    'HierarchicalTabICLClassifier',
    'HierarchicalTabICLRegressor',
    'train',
    'load_checkpoint',
    'generate_dataset',
    'generate_batch',
]
