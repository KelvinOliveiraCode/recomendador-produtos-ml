# BRIEF — recomendador-produtos-ml

## Projeto

Filtro colaborativo para e-commerce (area IA/ML): matriz usuario-item e recomendacao Top-N com precisao@K.

Stack: Python, implicit (ALS), pandas, NumPy, SciPy.

## Pasta de saida

`C:\Users\Kelvin\AppData\Local\Temp\opencode\github-staging\projects\recomendador-produtos-ml\`

## Arquivos criados

| Caminho | Conteudo |
|---|---|
| `pyproject.toml` | Declaracao do projeto, dependencias e `pythonpath`/`testpaths` do pytest |
| `src/recomendador_produtos_ml/__init__.py` | Export publico da API e versao |
| `src/recomendador_produtos_ml/data.py` | `Interactions`, `load_interactions`, `build_user_item_matrix` |
| `src/recomendador_produtos_ml/model.py` | `Recommender` (ALS, treino deterministico, Top-N) |
| `src/recomendador_produtos_ml/metrics.py` | `precision_at_k` |
| `tests/__init__.py` | Marcador de pacote de testes |
| `tests/conftest.py` | Fixtures compartilhadas |
| `tests/test_data.py` | Validacao de colunas, DataFrame vazio, formato/mapes da matriz |
| `tests/test_metrics.py` | Casos da precisao@K (perfeito, parcial, sem acertos, vazio, k=0, k>N) |
| `tests/test_model.py` | Treino, top-N exclui vistos, usuario desconhecido, nao treinado, precisao em holdout, determinismo |
| `README.md` | Documentacao bilíngue PT-BR + EN (titulo, descricao, como usar, testes, licenca MIT, autor) |
| `BRIEF.md` | Este resumo |

## Critérios de aceitação (do PLANO_50_PROJETOS.md)

- [x] Matriz usuario-item construída a partir de interacoes
- [x] Recomendacao Top-N (exclui itens ja vistos pelo usuario)
- [x] Metrica precisao@K implementada e testada
- [x] Suíte pytest passa (16/16)
- [x] README completo em PT-BR e EN
- [x] Codigo limpo, tipado, com docstrings, deterministico

## Comando de teste

```
cd C:\Users\Kelvin\AppData\Local\Temp\opencode\github-staging\projects\recomendador-produtos-ml
python -m pytest -q
```

## Resultado dos testes

16 passed, 0 failed (verificado por execucao real).

## Como verificar

1. Abra o terminal na pasta do projeto.
2. Confirme que `implicit` esta instalado: `python -c "import implicit; print(implicit.__version__)"`.
3. Rode `python -m pytest -q` — deve retornar `16 passed`.
4. O `README.md` e o `BRIEF.md` estao na raiz da pasta.

## Autor

Kelvin Oliveira
