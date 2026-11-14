# AdaSym: análise dos experimentos e diagnóstico do boost

**Pesquisa experimental de Ítalo Capuzzo em algoritmos de otimização para machine learning.**

A pesquisa combina descoberta simbólica, experimentação com redes neurais e análise de dados. Curvas de loss, varreduras de hiperparâmetros e contadores internos participam da identificação de padrões e da revisão das hipóteses sobre a regra de atualização.

## 1. Evolução da fórmula de atualização

A formulação evolui da expressão descoberta para uma regra com memória ao quadrado, piso relativo, memória direcional separada e peso temporal. A análise dos resultados acompanha esse desenvolvimento: investiga tanto o desempenho de treinamento quanto o mecanismo de amplificação por coordenada.

| Componente | Questão investigada |
| --- | --- |
| Memória direcional | Como a consistência recente dos gradientes participa do passo? |
| Piso v/c | Como limitar a amplificação produzida pela correção? |
| Coeficientes de m2 | Como a parametrização da memória afeta o objetivo? |
| Peso temporal | Como a participação da correção evolui durante o treinamento? |

## 2. Boost direcional e threshold

O AdaSym reduz o denominador da atualização de acordo com a memória de consistência dos gradientes. Esse mecanismo produz um boost element-wise, limitado pelo piso v/c. Na configuração examinada, o piso é v/4, com LeNet/Fashion-MNIST durante 150 épocas.

Os registros c4, c2 e full acompanham o mecanismo de boost. full está associado ao boost integral limitado pelo threshold; c4 e c2 acompanham faixas de boost parcial relacionadas a v/4 e v/2. c4 está incluído em c2.

Os registros são acumulados. Para examinar cada intervalo, utilizam-se diferenças sucessivas dos contadores. A composição de c2 é dividida em c4 e c2 fora de c4, evitando dupla contagem.

| Contador | Quantidade acumulada |
| --- | --- |
| c4 | 145,766,992,885 |
| c2 | 205,636,961,999 |
| c2 fora de c4 | 59,869,969,114 |
| full | 1,460,922,532,104 |

![Dinâmica dos contadores](threshold_dynamics.png)

## 3. Composição e evolução temporal

Dentro dos registros de c2, c4 representa **70.89%** e a faixa complementar representa **29.11%**. Essas proporções descrevem a composição interna de c2.

| Janela de épocas | c4 | c2 | full | c4 dentro de c2 |
| --- | --- | --- | --- | --- |
| 1–10 | 11,126,285,494 | 15,102,225,666 | 92,844,298,064 | 73.673% |
| 11–40 | 31,260,496,830 | 44,951,342,580 | 293,159,621,580 | 69.543% |
| 41–70 | 29,550,356,076 | 42,437,612,120 | 293,159,621,580 | 69.632% |
| 71–100 | 28,571,608,982 | 40,383,874,792 | 293,159,621,580 | 70.750% |
| 101–130 | 27,525,443,216 | 38,311,266,233 | 293,159,621,580 | 71.847% |
| 131–150 | 17,732,802,287 | 24,450,640,608 | 195,439,747,720 | 72.525% |

As janelas agregam as contagens antes de calcular a composição. As séries por intervalo e acumuladas permitem examinar a dinâmica do boost ao longo do treinamento. A janela móvel reúne até dez intervalos e descreve a evolução das faixas monitoradas.

![Composição por fase](threshold_phases.png)

![Contagens do boost](threshold_sensitivity.png)

## 4. Dinâmica da loss

As curvas são analisadas por trajetória e por grupo. A AUC normalizada utiliza a integral trapezoidal entre a primeira e a última posição registrada, dividida pelo comprimento desse intervalo. O resumo final considera as últimas 20 épocas.

| Grupo | Método | AUC normalizada | Loss média nas últimas 20 | DP temporal médio |
| --- | --- | --- | --- | --- |
| Fashion-MNIST / LeNet | AdamW | 0.133193 | 0.043519 | 0.003417 |
| Fashion-MNIST / LeNet | AdaSym | 0.118643 | 0.039634 | 0.003163 |
| Fashion-MNIST / LeNet | AdaSym nored | 0.115411 | 0.039071 | 0.007535 |
| Fashion-MNIST / LeNet | Controle LR 2× | 0.107722 | 0.042096 | 0.005114 |
| CIFAR-10 / ResNet-20 | AdamW | 0.077817 | 0.018017 | 0.004495 |
| CIFAR-10 / ResNet-20 | AdaSymSigmoid | 0.065551 | 0.015440 | 0.004120 |

A leitura conjunta da curva inteira e do trecho final permite distinguir aceleração inicial, qualidade tardia e variação temporal. A dispersão entre trajetórias é apresentada nas faixas do gráfico, enquanto o DP temporal descreve a variação dentro da janela final de cada trajetória.

![Diagnóstico das losses](loss_diagnostics.png)

Os resultados para diferentes alvos e quantidades de épocas consecutivas estão em `convergence_targets.csv`. O arquivo preserva o resultado de cada trajetória; o registro de época permanece vazio quando o alvo não é atingido no horizonte observado.

## 5. Sensibilidade aos hiperparâmetros

As varreduras permitem examinar a resposta do objetivo à learning rate e aos coeficientes de memória. As tabelas mantêm cada configuração da grade, possibilitando análise além do ponto de mínimo.

![Exploração dos hiperparâmetros](hyperparameter_diagnostics.png)

## 6. Síntese da análise

Os dados conectam a evolução da fórmula ao diagnóstico do mecanismo e à comparação experimental. Os contadores mostram a dinâmica das faixas de boost; as curvas de loss descrevem o comportamento de treinamento; as varreduras mostram a resposta aos hiperparâmetros. A combinação dessas perspectivas caracteriza a ciência de dados como metodologia da pesquisa AdaSym.

## Reprodução e tabelas

```bash
python -m pip install -e ".[plots,docs]"
python scripts/analyze_research.py
```

Os resultados incluem CSVs por época e trajetória, figuras PNG/SVG, manifesto com hashes dos dados e uma cópia HTML com figuras incorporadas. O notebook `notebooks/threshold-diagnostics.ipynb` apresenta o diagnóstico executável.
