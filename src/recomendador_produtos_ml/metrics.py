"""Metricas de avaliacao de recomendacao."""
from __future__ import annotations

from typing import Iterable, List, Set


def precision_at_k(
    recommended: Iterable[int],
    relevant: Iterable[int],
    k: int,
) -> float:
    """Precisao@K: fracao dos K itens recomendados que sao relevantes.

    Args:
        recommended: itens ordenados pelo modelo (maior relevancia primeiro).
        relevant: conjunto de itens considerados relevantes para o usuario.
        k: numero de recomendacoes do topo a avaliar.

    Returns:
        Precisao@K entre 0.0 e 1.0. Retorna 0.0 quando k <= 0 ou quando nao ha
        recomendacoes para avaliar.
    """
    if k <= 0:
        return 0.0
    relevant_set: Set[int] = set(relevant)
    top_k: List[int] = list(recommended)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(top_k)
