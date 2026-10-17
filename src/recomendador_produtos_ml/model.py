"""Modelo de filtro colaborativo baseado em ALS (implicit)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from .data import Interactions, build_user_item_matrix


class Recommender:
    """Recomendador Top-N por filtragem colaborativa (ALS).

    Encapsula o ``implicit.als.AlternatingLeastSquares`` e expoe uma interface
    deterministica (``random_state`` fixo) para treino e recomendacao.

    Atributos publicos apos ``fit``:
        user_item_matrix: matriz esparsa usuario-item (n_usuarios x n_itens).
        user_to_idx / item_to_idx: mapas de id original para indice interno.
        idx_to_user / idx_to_item: listas indexadas por indice interno.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.1,
        iterations: int = 20,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )
        self.user_to_idx: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_user: List[int] = []
        self.idx_to_item: List[int] = []
        self.user_item_matrix: csr_matrix | None = None
        self._fitted: bool = False

    def fit(self, interactions: Interactions) -> "Recommender":
        """Treina o modelo a partir das interacoes usuario-item.

        Args:
            interactions: objeto Interactions com usuarios, itens e avaliacoes.

        Returns:
            O proprio recomendador (encadeamento de chamadas).
        """
        (
            matrix,
            self.user_to_idx,
            self.item_to_idx,
            self.idx_to_user,
            self.idx_to_item,
        ) = build_user_item_matrix(interactions)
        self.user_item_matrix = matrix
        # O implicit espera a matriz usuario-item (usuarios x itens) no fit.
        # BLAS com 1 thread garante treino deterministico.
        with threadpool_limits(limits=1, user_api="blas"):
            self.model.fit(matrix)
        self._fitted = True
        return self

    def recommend(
        self, user_id: int, n: int = 10
    ) -> List[Tuple[int, float]]:
        """Retorna os top-N itens recomendados para um usuario.

        Itens ja interagidos pelo usuario sao filtrados (nao sao recomendados).

        Args:
            user_id: identificador do usuario (deve estar no treino).
            n: numero maximimo de recomendacoes a retornar.

        Returns:
            Lista de pares (item_id, pontuacao), ordenada por pontuacao
            decrescente.

        Raises:
            RuntimeError: se o modelo nao foi treinado com fit().
            ValueError: se o usuario nao estiver no treino.
        """
        if not self._fitted:
            raise RuntimeError("O modelo precisa ser treinado com fit() antes de recomendar.")
        if user_id not in self.user_to_idx:
            raise ValueError(f"Usuario {user_id} nao esta no treino.")
        user_idx = self.user_to_idx[user_id]
        seen = set(int(i) for i in self.user_item_matrix[user_idx].indices.tolist())
        # Pontuacao deterministica: produto escalar dos fatores do usuario com
        # todos os fatores de item (mesma logica do implicit, com controle total).
        scores = self.model.user_factors[user_idx] @ self.model.item_factors.T
        scores = np.asarray(scores, dtype=float)
        for item_idx in seen:
            scores[item_idx] = -np.inf
        # Ordenacao estavel por pontuacao decrescente; desempate pelo indice.
        ordem = np.argsort(-scores, kind="stable")
        top_idx = [int(i) for i in ordem if int(i) not in seen][:n]
        return [
            (int(self.idx_to_item[i]), float(scores[i])) for i in top_idx
        ]

    def recommend_items(self, user_id: int, n: int = 10) -> List[int]:
        """Retorna apenas os IDs dos top-N itens recomendados.

        Args:
            user_id: identificador do usuario (deve estar no treino).
            n: numero maximimo de recomendacoes a retornar.

        Returns:
            Lista de IDs de itens recomendados.
        """
        return [item_id for item_id, _ in self.recommend(user_id, n=n)]
