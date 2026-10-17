"""Testes do modulo de dados (matriz usuario-item)."""
from __future__ import annotations

import pandas as pd
import pytest

from recomendador_produtos_ml import Interactions, build_user_item_matrix, load_interactions


def test_load_interactions_valida_colunas() -> None:
    df = pd.DataFrame({"usuario": [1], "item": [10], "nota": [1.0]})
    with pytest.raises(ValueError, match="Colunas ausentes"):
        load_interactions(df)


def test_load_interactions_rejeita_vazio() -> None:
    df = pd.DataFrame(columns=["user_id", "item_id", "rating"])
    with pytest.raises(ValueError, match="vazio"):
        load_interactions(df)


def test_build_matrix_shapes_e_mapes(interactions: Interactions) -> None:
    matrix, user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_user_item_matrix(
        interactions
    )
    assert matrix.shape == (4, 5)
    assert set(user_to_idx) == {1, 2, 3, 4}
    assert set(item_to_idx) == {10, 20, 30, 40, 50}
    assert idx_to_user == [1, 2, 3, 4]
    assert idx_to_item == [10, 20, 30, 40, 50]
    # round-trip dos mapes id <-> indice
    for uid, uidx in user_to_idx.items():
        assert idx_to_user[uidx] == uid
    for iid, iidx in item_to_idx.items():
        assert idx_to_item[iidx] == iid
