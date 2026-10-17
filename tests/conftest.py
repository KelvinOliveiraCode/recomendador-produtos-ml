"""Fixtures compartilhadas para os testes."""
from __future__ import annotations

import pandas as pd
import pytest

from recomendador_produtos_ml import Interactions, Recommender
from recomendador_produtos_ml.data import load_interactions


@pytest.fixture
def interactions_df() -> pd.DataFrame:
    """Interacoes sinteticas e determinísticas (4 usuarios x 5 itens)."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "item_id": [10, 20, 30, 10, 20, 40, 20, 30, 50, 10, 30, 50],
            "rating": [1.0] * 12,
        }
    )


@pytest.fixture
def interactions(interactions_df: pd.DataFrame) -> Interactions:
    return load_interactions(interactions_df)


@pytest.fixture
def recommender(interactions: Interactions) -> Recommender:
    return Recommender(factors=8, iterations=5, random_state=7).fit(interactions)
