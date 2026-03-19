# CrediFacil - Previsao de Aprovacao de Emprestimo

Este projeto utiliza Regressao Logistica para automatizar a decisao de aprovacao de emprestimos pessoais com base no perfil do cliente.

## Objetivo
Classificar se um pedido de emprestimo deve ser aprovado (1) ou negado (0).

## Etapas do Projeto
1. Carregamento dos dados e analise inicial.
2. Analise exploratoria com distribuicao das variaveis.
3. Divisao dos dados em conjuntos de treino e teste.
4. Treinamento do modelo de Regressao Logistica.
5. Avaliacao atraves de matriz de confusao, acuracia, precisao e recall.
6. Interpretacao dos coeficientes para identificar variaveis decisivas.

## Requisitos
Python 3 e bibliotecas:
- pandas
- matplotlib
- seaborn
- scikit-learn

## Como usar
1. Mantenha o arquivo dataset_emprestimo_aprovacao.csv na mesma pasta.
2. Execute o script para ver as metricas de desempenho e a importancia de cada variavel (renda, score e dividas).