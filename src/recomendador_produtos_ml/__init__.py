"""recomendador-produtos-ml — filtro colaborativo para e-commerce."""

from .data import Interactions, build_user_item_matrix, load_interactions
from .metrics import precision_at_k
from .model import Recommender

__all__ = [
    "Interactions",
    "Recommender",
    "build_user_item_matrix",
    "load_interactions",
    "precision_at_k",
]

__version__ = "0.1.0"
