"""Testes do modelo de recomendacao."""
from __future__ import annotations

import pandas as pd
import pytest

from recomendador_produtos_ml import Interactions, Recommender, precision_at_k
from recomendador_produtos_ml.data import load_interactions


def test_treina_sem_erro(recommender: Recommender) -> None:
    assert recommender._fitted is True


def test_recomenda_top_n(recommender: Recommender) -> None:
    resultado = recommender.recommend_items(user_id=1, n=5)
    assert len(resultado) <= 5
    assert all(isinstance(i, int) for i in resultado)
    # nao recomenda itens ja vistos pelo usuario 1 (10, 20, 30)
    assert not set(resultado) & {10, 20, 30}


def test_recomenda_usuario_desconhecido_erra(recommender: Recommender) -> None:
    with pytest.raises(ValueError, match="nao esta no treino"):
        recommender.recommend_items(user_id=99, n=5)


def test_nao_treinado_erra() -> None:
    modelo = Recommender()
    with pytest.raises(RuntimeError, match="treinado"):
        modelo.recommend_items(user_id=1, n=5)


def test_precisao_holdout_esta_no_intervalo(
    interactions_df: pd.DataFrame,
) -> None:
    """Precisao@K em holdout por usuario: treina sem um item, avalia esse item."""
    treino = interactions_df.groupby("user_id").head(2)
    holdout = interactions_df.groupby("user_id").tail(1)
    modelo = Recommender(factors=8, iterations=5, random_state=7).fit(
        load_interactions(treino)
    )
    soma = 0.0
    n = 0
    for usuario in holdout["user_id"].unique():
        relevantes = set(
            holdout[holdout["user_id"] == usuario]["item_id"].tolist()
        )
        recomendados = modelo.recommend_items(usuario, n=5)
        soma += precision_at_k(recomendados, relevantes, k=5)
        n += 1
    assert n == 4
    media = soma / n
    assert 0.0 <= media <= 1.0


def test_determinismo(interactions: Interactions) -> None:
    m1 = Recommender(factors=8, iterations=5, random_state=7).fit(interactions)
    m2 = Recommender(factors=8, iterations=5, random_state=7).fit(interactions)
    assert m1.recommend_items(1, n=5) == m2.recommend_items(1, n=5)
