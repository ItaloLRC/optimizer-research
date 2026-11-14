# Mapa dos experimentos

| Experimento | Registros | Configuração e análise |
|---|---|---|
| Fashion-MNIST | losses_FashionMNIST.npy | 100 épocas; AdaSym, AdaSymSigmoid e AdamW |
| Dinâmica em LeNet | AdamW_losses.npy, AdaSym_losses.npy, AdaSym_loss_fmnist_nored.npy | Três trajetórias de 150 épocas por arquivo |
| CIFAR-10 | AdamW_losses_cifar10.npy, AdaSymSig_losses_cifar10.npy | ResNet-20, 150 épocas |
| Controle de LR | Adam_losses2x.npy | LR 2×, 150 épocas |
| Memória | loss_matrix.npy, loss_matrix (1).npy | Duas grades de 10 × 70 pontos |
| Learning rate | lrs_aucloss.npy, lrs_aucloss_adamW.npy | Varreduras de 27 pontos |
| Learning rate de AdaSym | lrs_AdaSym_aucloss.npy | 80 registros do objetivo |
| Sigmoide | lim_aucloss.npy | 50 registros do objetivo |
| Boost por coordenada | AdaSym_fminist_c4.npy, AdaSym_fminist_c2.npy, AdaSym_fminist_full.npy | LeNet/Fashion-MNIST, 150 épocas, piso v/4; c4 incluído em c2 |

O [relatório](../results/research/report.md) reúne o diagnóstico dos contadores, a dinâmica da loss e as varreduras. O [esquema](../data/experiment_schema.json) descreve os registros de boost.
