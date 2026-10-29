# Experimento novo em MLP sintético

Classificação não linear, 900 amostras de treino, 250 de validação e 250 de teste. MLP tanh 12–24–2, minibatch 64. Valores abaixo são média ± desvio padrão amostral entre 5 inicializações. O conjunto de teste é o mesmo nas 5 inicializações.

Épocas: 35. Fórmula selecionada na triagem: `sqrt(q)`. A triagem usou outras sementes e outras amostras.

| Método | LR congelada | Test loss | Test accuracy | AUC val média por época |
|---|---:|---:|---:|---:|
| adamw_tuned | 0.003 | 0.49164 ± 0.01840 | 0.76400 ± 0.02668 | 0.57811 ± 0.00567 |
| adamw_scaled_lr | 0.006 | 0.36677 ± 0.03558 | 0.83360 ± 0.01345 | 0.52476 ± 0.01715 |
| historical | 0.003 | 0.46001 ± 0.02915 | 0.79040 ± 0.03182 | 0.56906 ± 0.00774 |
| adasym_linear | 0.003 | 0.47284 ± 0.02673 | 0.78000 ± 0.03187 | 0.57171 ± 0.00712 |
| symbolic_selected | 0.003 | 0.44138 ± 0.03437 | 0.79760 ± 0.03067 | 0.56213 ± 0.01029 |

## Limites da interpretação

Este teste verifica a implementação e fornece evidência limitada sobre esta tarefa sintética. A seleção simbólica consumiu 15 configurações, enquanto cada baseline usou 3. A comparação da melhor família simbólica contra uma baseline inclui esse orçamento adicional e não prova superioridade em orçamento igual. A confirmação usa apenas uma tarefa nova, 5 inicializações e amostras do mesmo gerador. Não substitui Fashion-MNIST, CIFAR-10 ou avaliação de transferência entre domínios.

Os modelos de teste usam o checkpoint de menor loss de validação. As configurações não são escolhidas pelo teste. O desvio padrão descreve variação entre inicializações, não incerteza entre datasets nem um intervalo de confiança.

Os métodos históricos são reconstruções novas com momentos explicitados. Não reproduzem o treinamento original ausente. A formulação linear com janela comum constitui a hipótese principal, mesmo se outra transformação vencer a triagem. Não se escolhe uma ideia final com base neste único experimento.
