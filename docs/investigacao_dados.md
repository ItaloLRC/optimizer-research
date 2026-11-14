# Análise de dados na pesquisa AdaSym

Pesquisa experimental de **Ítalo Capuzzo** em algoritmos de otimização para machine learning.

## Formulação e evidência experimental

A investigação reúne descoberta simbólica, comparação de curvas de loss, exploração de hiperparâmetros e diagnóstico do boost element-wise. A análise dos padrões participa da evolução da regra de atualização: memória ao quadrado, piso relativo, memória direcional separada e ponderação temporal.

## Diagnóstico do mecanismo de boost

Os registros c4/c2/full acompanham o treinamento de LeNet em Fashion-MNIST durante 150 épocas, com piso v/4. full está associado ao boost integral limitado pelo threshold. c4 e c2 acompanham faixas de boost parcial relacionadas a v/4 e v/2, com c4 incluído em c2.

As séries acumuladas são convertidas em incrementos por intervalo. A partição de c2 em c4 e c2 fora de c4 permite examinar sua composição sem dupla contagem. As janelas utilizam somas dos contadores antes do cálculo das proporções internas de c2.

## Análise dos resultados

As curvas de loss são examinadas por trajetória e por grupo, incluindo AUC normalizada e comportamento no trecho final. As varreduras preservam o objetivo de cada configuração. Em conjunto, essas análises relacionam o funcionamento do otimizador ao comportamento experimental.

## Materiais

- [Relatório de análise](../results/research/report.md)
- [Notebook executável](../notebooks/threshold-diagnostics.ipynb)
- [Contadores por época](../results/research/threshold_by_epoch.csv)
- [Esquema dos registros](../data/experiment_schema.json)

O comando `python scripts/analyze_research.py` gera as tabelas, figuras e o relatório. O manifesto registra os hashes dos dados e os parâmetros utilizados.
