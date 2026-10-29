# Convergência e oscilação da loss no final do treino

A comparação considera as curvas salvas de 150 épocas. O tempo até o alvo é a primeira época de uma sequência de cinco épocas em que a média das três trajetórias permanece em loss ≤ 0,1. A medida usa épocas de treinamento, sem estimar tempo de execução.

A variabilidade final é a média dos desvios padrão temporais, calculados separadamente em cada trajetória nas últimas 20 épocas, com ddof=1. Ela inclui qualquer tendência residual desse trecho. A medida descreve a oscilação da loss e não equivale à dispersão entre sementes ou a uma garantia de estabilidade do otimizador.

| Experimento | Método | Época até o alvo na curva média | Loss média nas últimas 20 épocas | DP temporal médio nas últimas 20 épocas |
|---|---|---:|---:|---:|
| Fashion-MNIST / LeNet | AdamW | 73 | 0.043519 | 0.003417 |
| Fashion-MNIST / LeNet | AdaSym | 63 | 0.039634 | 0.003163 |
| CIFAR-10 / ResNet-20 | AdamW | 22 | 0.018017 | 0.004495 |
| CIFAR-10 / ResNet-20 | AdaSymSigmoid | 16 | 0.015440 | 0.004120 |
| Grupo LeNet, variante sem redução temporal | AdaSym nored | 60 | 0.039071 | 0.007535 |
| Grupo LeNet, rótulo de dataset divergente | AdamW LR 2× | 54 | 0.042096 | 0.005114 |

## Interpretação

No par Fashion-MNIST/LeNet descrito com LR 1e-3, AdaSym atinge o alvo na época 63 e AdamW na 73. No par CIFAR-10/ResNet-20, ambos com LR 1e-3, AdaSymSigmoid atinge o alvo na época 16 e AdamW na 22. As variantes AdaSym dessas duas comparações apresentam loss média menor e DP temporal ligeiramente menor nas últimas 20 épocas.

Essas observações apoiam a hipótese de que AdaSym acelera a convergência em número de épocas com oscilação final da loss comparável à de AdamW, nas configurações avaliadas. Elas não demonstram equivalência estatística de estabilidade nem superioridade em tarefas não avaliadas.

A conclusão tem exceções importantes: o controle AdamW com LR 2× alcança o alvo antes de AdaSym, e a variante AdaSym nored apresenta DP temporal maior no final. A alegação de aceleração se refere aos pares descritos com a mesma LR; o comportamento depende da variante e do ajuste dos hiperparâmetros.

O alvo de 0,1, as cinco épocas consecutivas e a janela final de 20 épocas foram escolhidos após observar as curvas, para produzir uma descrição auditável. Não constituem um protocolo pré-registrado. Os IDs de sementes e o split da loss não acompanham os arquivos; há ainda uma ambiguidade de scheduler no grupo LeNet e de dataset no controle LR 2×, detalhada no mapa de experimentos. Uma comparação confirmatória deve congelar essas escolhas antes dos próximos testes.
