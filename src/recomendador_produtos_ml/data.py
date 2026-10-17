"""Carregamento e preparacao de interacoes usuario-item."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Interactions:
    """Interacoes usuario-item brutas."""

    users: List[int]
    items: List[int]
    ratings: List[float]


def load_interactions(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
) -> Interactions:
    """Valida e extrai interacoes de um DataFrame.

    Args:
        df: DataFrame com pelo menos as colunas de usuario, item e avaliacao.
        user_col: nome da coluna de identificador do usuario.
        item_col: nome da coluna de identificador do item.
        rating_col: nome da coluna de avaliacao/implicit feedback.

    Returns:
        Objeto Interactions com listas de usuarios, itens e avaliacoes.

    Raises:
        ValueError: se colunas obrigatorias faltarem ou o DataFrame estiver vazio.
    """
    missing = {user_col, item_col, rating_col} - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no DataFrame: {sorted(missing)}")
    if len(df) == 0:
        raise ValueError("O DataFrame de interacoes nao pode estar vazio.")
    users = df[user_col].tolist()
    items = df[item_col].tolist()
    ratings = df[rating_col].astype(float).tolist()
    return Interactions(users=users, items=items, ratings=ratings)


def build_user_item_matrix(
    interactions: Interactions,
) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int], List[int], List[int]]:
    """Constroi a matriz esparsa usuario-item e os mapas de id <-> indice.

    Args:
        interactions: objeto Interactions com usuarios, itens e avaliacoes.

    Returns:
        matrix: csr_matrix (n_usuarios x n_itens) com as avaliacoes.
        user_to_idx: mapeamento user_id -> linha.
        item_to_idx: mapeamento item_id -> coluna.
        idx_to_user: lista indexada por linha -> user_id.
        idx_to_item: lista indexada por coluna -> item_id.
    """
    user_ids = sorted({int(u) for u in interactions.users})
    item_ids = sorted({int(i) for i in interactions.items})
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for user, item, rating in zip(
        interactions.users, interactions.items, interactions.ratings
    ):
        rows.append(user_to_idx[int(user)])
        cols.append(item_to_idx[int(item)])
        vals.append(float(rating))

    matrix = csr_matrix(
        (vals, (rows, cols)), shape=(len(user_ids), len(item_ids)), dtype=float
    )
    return matrix, user_to_idx, item_to_idx, user_ids, item_ids
