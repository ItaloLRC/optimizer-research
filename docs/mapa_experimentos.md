# Mapa dos testes recebidos

Transcrição organizada de `data/original/metadata.txt`, cruzada com shapes e slides. Campos não informados permanecem desconhecidos. Os nomes preservam a grafia original, incluindo `fminist`.

| Arquivos | Descrição do metadata / slides | O que os dados confirmam | Pontos a resolver |
|---|---|---|---|
| `losses_FashionMNIST.npy` | Fashion-MNIST, 100 épocas, AdaSym / AdaSymSig / AdamW. Slides 8 e 9 | 3 × 100; ordem inferida AdaSym, AdaSymSigmoid, AdamW pelos valores iniciais e finais | Split da loss, seeds, configuração e ordem pelo código original |
| `AdamW_losses.npy`, `AdaSym_losses.npy` | LeNet, Fashion-MNIST, 150 épocas, LR 1e-3, sem redução por sigmoide ou exponencial em um trecho; outro trecho cita v/4, (0,75; 0,35), sigmoide por época | Cada arquivo tem 3 × 150. Slides mostram média e DP | Qual variante pertence exatamente a cada arquivo; recorrência e significado dos betas |
| `AdaSym_loss_fmnist_nored.npy` | Nome sugere ausência de redução temporal, compatível com parte da descrição acima | 3 × 150 | Ligação exata com os trechos de metadata e o baseline correspondente |
| `AdamW_losses_cifar10.npy`, `AdaSymSig_losses_cifar10.npy` | CIFAR-10, ResNet-20, 150 épocas, LR 1e-3, WD 1e-2, sigmoide em torno da época 15, piso v/8. Slides 12 e 13 | Cada arquivo tem 3 × 150 e curvas coerentes com o gráfico de média | Acurácia, split, scheduler, seeds, arquitetura exata e código |
| `Adam_losses2x.npy` | Metadata chama AdamW com LR 2×, LeNet, MNIST, 150 épocas. Slides 14 e 15 dizem Fashion-MNIST | 3 × 150 e curva coerente com o controle AdamW 2× dos slides | Conflito MNIST/Fashion-MNIST e nome Adam/AdamW |
| `loss_matrix.npy` | Varredura para m2. beta3 = 10^linspace(−0,3; 0; 10), beta4 = 10^linspace(−3; 0; 70). Slide 18 | 10 × 70, 70 NaNs na última linha. Mínimo finito coincide com o destaque do slide | Definição da recorrência, fitness e origem dos NaNs |
| `loss_matrix (1).npy` | Segunda varredura. beta3 = 10^linspace(−0,3; −0,01; 10), beta4 = 10^linspace(−1,5; 0; 70). Slide 19 | 10 × 70, sem NaN. Mínimo coincide com o destaque do slide | Mesmas lacunas da recorrência e do fitness |
| `lrs_AdaSym_aucloss.npy` | 300 épocas, 10% de Fashion-MNIST, AUC de loss, WD 1e-2, valores (0,75; 0,35) | Vetor com 80 posições e valores muito altos em parte da faixa | A grade de 80 LRs e quais valores representam divergência |
| `lrs_aucloss.npy`, `lrs_aucloss_adamW.npy` | Uma grade concatenada 1e-4…9e-4, 1e-3…9e-3, 1e-2…9e-2; trecho próximo cita sigmoide(15−época), piso v/8 | Cada vetor tem 27 posições, compatível com o tamanho da grade | Associação exata entre configuração, fitness e cada vetor; o slide 17 usa outra descrição de métrica |
| `lim_aucloss.npy` | Ponto médio da sigmoide entre 0 e 50, LeNet, 20% Fashion-MNIST, LR 1e-3, WD 1e-2, 150 épocas | Vetor com 50 posições | Valores exatos do eixo, repetições e definição de AUC |
| `AdaSym_fminist_c2.npy`, `AdaSym_fminist_c4.npy`, `AdaSym_fminist_full.npy` | LeNet, Fashion-MNIST, piso v/4; contagens para ≤v/2, ≤v/4 e total. Metadata diz 1550 épocas. Slide 16 mostra 150 | Três vetores inteiros monotônicos de tamanho 150; razões acumuladas coincidem com o slide | Unidade de contagem, frequência e correção do número de épocas |

O metadata também menciona `div_aucloss`, uma varredura de divisor de 2 a 20 com 150 épocas, LeNet e 20% de Fashion-MNIST. Esse arquivo não acompanha a pasta recebida, portanto não há como recalcular esse teste.

Os exemplos de código preservados em metadata esclarecem a intenção do denominador:

```python
alpha = torch.sigmoid(torch.tensor(15 - cur_epoch))
val = torch.clamp(v_hat - s_hat.square() * alpha, min=v_hat/8)
```

Eles não definem como `v_hat` e `s_hat` são calculados, nem epsilon, bias correction e atualização dos parâmetros. Por isso, a implementação nova explicita essas decisões em vez de atribuí-las ao histórico.
