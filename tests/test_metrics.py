"""Testes da metrica precisao@K."""
from __future__ import annotations

from recomendador_produtos_ml.metrics import precision_at_k


def test_precision_at_k_perfeito() -> None:
    assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_precision_at_k_parcial() -> None:
    assert precision_at_k([1, 2, 3, 4], {2, 4}, k=4) == 0.5


def test_precision_at_k_avalia_apenas_o_top_k() -> None:
    assert precision_at_k([1, 2, 3], {2}, k=2) == 0.5


def test_precision_at_k_sem_acertos() -> None:
    assert precision_at_k([1, 2, 3], {9, 10}, k=3) == 0.0


def test_precision_at_k_sem_recomendacoes() -> None:
    assert precision_at_k([], {1}, k=5) == 0.0


def test_precision_at_k_k_zero() -> None:
    assert precision_at_k([1, 2, 3], {1}, k=0) == 0.0


def test_precision_at_k_k_maior_que_disponivel() -> None:
    assert precision_at_k([1, 2], {1, 2}, k=10) == 1.0
