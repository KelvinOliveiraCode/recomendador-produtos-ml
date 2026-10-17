# recomendador-produtos-ml

Filtro colaborativo para e-commerce com recomendacao Top-N e precisao@K.
Collaborative filtering for e-commerce with Top-N recommendation and precision@K.

## Sobre / About

**Portugues.** Pacote Python de filtragem colaborativa baseado em Alternating Least Squares (ALS) pela biblioteca `implicit`. Recebe interacoes usuario-item (visualizacoes, compras ou avaliacoes), constroi a matriz esparsa usuario-item, treina o modelo e devolve os N itens mais relevantes para cada usuario, excluindo o que ja foi visto. Inclui a metrica precisao@K para avaliar as recomendacoes.

**Ingles.** Python package for collaborative filtering based on Alternating Least Squares (ALS) via the `implicit` library. It takes user-item interactions (views, purchases, or ratings), builds the sparse user-item matrix, trains the model, and returns the N most relevant items for each user, excluding items already seen. Includes the precision@K metric to evaluate recommendations.

## Stack

Python 3.10+, pandas, NumPy, SciPy, implicit.

## Como usar / How to use

Instale o pacote em modo editavel:

```
pip install -e .
```

Treine o modelo e gere recomendacoes:

```python
from pandas import DataFrame
from recomendador_produtos_ml import (
    Interactions,
    Recommender,
    precision_at_k,
    load_interactions,
)

df = DataFrame(
    {
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "item_id": [10, 20, 30, 10, 20, 40, 20, 30, 50, 10, 30, 50],
        "rating": [1.0] * 12,
    }
)
interacoes = load_interactions(df)

modelo = Recommender(factors=8, iterations=5, random_state=7)
modelo.fit(interacoes)

# Top-5 itens recomendados para o usuario 1 (sem os ja vistos)
recomendados = modelo.recommend_items(1, n=5)
print(recomendados)

# Precisao@K contra os itens relevantes do usuario
precisao = precision_at_k(recomendados, {20, 30, 40}, k=5)
print(precisao)
```

`Recommender` e deterministico: mesmo `random_state` e mesmos dados produzem as mesmas recomendacoes.

## Testes / Tests

Execute a suíte com pytest:

```
python -m pytest -q
```

Todos os testes devem passar. Nao ha testes que dependam de rede ou de dados externos.

## Licenca / License

MIT License

Copyright (c) 2026 Kelvin Oliveira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Autor / Author

Kelvin Oliveira
