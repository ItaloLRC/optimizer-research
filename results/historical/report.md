# Auditoria dos resultados originais

Cada linha das matrizes 3 × 150 é tratada como trajetória sem identidade confirmada. A tabela agrega essas três trajetórias descritivamente. Não se presume que sejam sementes independentes ou que a loss seja de validação. Arquivos e hashes constam em `inventory.json`. AUC usa espaçamento de uma época, começando na primeira loss registrada. O código do cálculo original de AUC não foi fornecido.

| Arquivo | Média das últimas 10 posições | Média de todas as posições |
|---|---:|---:|
| AdaSymSig_losses_cifar10.npy | 0.015382 | 0.069594 |
| AdaSym_loss_fmnist_nored.npy | 0.038083 | 0.117264 |
| AdaSym_losses.npy | 0.038349 | 0.120462 |
| AdamW_losses.npy | 0.041992 | 0.135194 |
| AdamW_losses_cifar10.npy | 0.017411 | 0.081801 |
| Adam_losses2x.npy | 0.041124 | 0.109575 |

## Cruzamento com o PowerPoint

Os gráficos dos slides 10, 12 e 14 identificam médias e desvios padrão, coerentes com três trajetórias por arquivo 3 × 150. Isso sustenta a interpretação como repetições, mas não identifica sementes ou independência. O gráfico do slide 16 explicita contadores acumulados, reforçando essa leitura dos arquivos c2/c4/full.

Para `losses_FashionMNIST.npy` (3 × 100), a correspondência dos valores iniciais e finais com os gráficos dos slides 8 e 9 permite inferir a ordem abaixo. É uma inferência visual verificável, ainda sem o programa que gerou o gráfico. Essas linhas representam métodos diferentes e não três sementes do mesmo método.

| Linha (índice zero) | Método inferido | Primeiro valor | Último valor | Média registrada |
|---|---|---:|---:|---:|
| 0 | AdaSym | 0.553933 | 0.278333 | 0.291404 |
| 1 | AdaSymSigmoid | 0.550282 | 0.262625 | 0.287134 |
| 2 | AdamW | 0.595231 | 0.275254 | 0.302175 |

Os gráficos dos slides 18 e 19 indicam mínimos em (0,5843; 0,7406) e (0,6744; 0,4062). A reconstrução dos eixos pelo metadata e os mínimos numéricos abaixo coincidem com essas marcações. Essa concordância identifica as duas varreduras, sem resolver o significado de beta4.

## Inconsistências e limites

- `loss_matrix.npy` inclui betas iguais a 1 na grade declarada, inválidos para uma EMA normalizada com bias correction. A presença de NaN coincide com a última linha, mas falta código para estabelecer a causa.
- `lrs_AdaSym_aucloss.npy` tem 80 valores. A grade com 27 LRs em metadata não permite reconstruir seu eixo. Não se atribui uma LR ótima a esse arquivo.
- O slide 17 rotula sua métrica como média das últimas 15 losses e mostra LRs até 1e-2. Os vetores de LR em metadata são descritos como AUC e incluem valores até 9e-2. Não se presume que sejam a fonte exata desse gráfico sem o pós-processamento original.
- `metadata.txt` diz 1550 épocas para os contadores. Os três arquivos correspondentes têm 150 posições, compatíveis com o eixo do slide 16. Preserva-se a divergência textual no registro original.
- `Adam_losses2x.npy` é chamado AdamW 2× no texto e nos gráficos, mas o nome diz Adam. O dataset aparece como MNIST em metadata e Fashion-MNIST nos slides. A comparação não é rotulada como controlada.
- A sigmoide usa ponto de transição 5 no exemplo visual e 15 em metadata. As recorrências de m2, bias correction, epsilon e scheduler completo estão ausentes.
- Ausência de acurácia, separação treino/validação/teste, sementes, checkpoints, tempos, versões e logs da busca impede concluir superioridade de generalização ou reconstruir a regressão simbólica histórica.

## Grade de betas

| Arquivo | Não finitos | Menor valor finito | Índice mínimo | Betas inferidos por metadata |
|---|---:|---:|---|---|
| loss_matrix.npy | 70 | 15.459559 | (2, 66) | (0.584341, 0.740568) |
| loss_matrix (1).npy | 0 | 14.534201 | (4, 51) | (0.674355, 0.406159) |

Esses mínimos são observações de uma varredura, sem repetições identificadas. Não constituem hiperparâmetros confirmados. O papel de beta4 não é definido e não equivale automaticamente ao complemento 1−beta3.

## Contadores

- `c2`: razão acumulada final = 14.0758%. Se os arquivos registram contadores acumulados compatíveis, a razão por época deve usar diferenças sucessivas do numerador e denominador, não a razão acumulada em cada época.
- `c4`: razão acumulada final = 9.9777%. Se os arquivos registram contadores acumulados compatíveis, a razão por época deve usar diferenças sucessivas do numerador e denominador, não a razão acumulada em cada época.

As curvas registradas sugerem diferenças na dinâmica de treinamento. Os dados disponíveis não sustentam a afirmação de que um otimizador generaliza melhor nem separam o efeito da janela do efeito de uma LR maior.
