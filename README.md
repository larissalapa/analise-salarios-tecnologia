# Análise Preditiva de Salários em Tecnologia para a matéria de Data Science

Este projeto tem como objetivo prever se um salário em tecnologia é acima ou abaixo da mediana, utilizando técnicas de Machine Learning.

## Descrição

- Dataset com 5000 registros de salários em tecnologia.
- Pré-processamento:
  - Remoção de duplicados
  - Tratamento de valores ausentes
  - Criação da variável alvo (`salario_alto`)
  - Codificação de variáveis categóricas
  - Normalização de variáveis numéricas
  - Divisão em treino e teste
- Modelos utilizados:
  - K-Nearest Neighbors (KNN)
  - Regressão Logística
  - Árvore de Decisão
  - Random Forest
- Avaliação: Acurácia, Precisão, Recall, F1-Score e Matrizes de Confusão.
- Análise de importância das features (Random Forest).

## Fonte dos Dados

Os dados foram obtidos a partir do dataset **[“Global Technology Salary Data”](https://www.kaggle.com/datasets/ruchi798/global-tech-salaries)** disponível no Kaggle.

## Como rodar

1. Clone o repositório:
```bash
git clone https://github.com/SEU_USUARIO/nome-do-repositorio.git
