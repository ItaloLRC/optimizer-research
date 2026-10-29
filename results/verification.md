# Verificação realizada

Data: 29 de outubro de 2026. Ambiente local Windows, Python 3.12.14. Experimento sintético executado com NumPy 2.3.5. Testes da regra executados com NumPy 2.5.4 e PyTorch 2.15.0+cpu.

## Testes da regra

Dez testes passaram, sem testes ignorados:

- Limites alpha=0 e c=1 comparados com recorrência AdamW independente.
- Estatística distingue sinal constante, cancelamento de sinal e variação de magnitude.
- Limites de ganho para todas as seis regras em gradientes aleatórios e nulos.
- Gradientes inválidos não alteram parâmetros e gradientes ausentes pulam a atualização.
- Simetria por inversão de sinal e reescala aproximada dos gradientes.
- Gradiente do MLP comparado com diferenças finitas em todos os tensores.
- Sigmoide exige progresso explícito e reduz o ganho no final.
- Retomada de estado PyTorch reproduz o passo seguinte.
- Paridade NumPy/PyTorch em 6 regras, 2 schedules e 25 passos.
- Alpha=0 comparado com `torch.optim.AdamW` nativo em precisão dupla.

## Execuções empíricas

A triagem sintética executou 30 treinamentos de candidatos simbólicos e 24 treinamentos para quatro controles/famílias. A confirmação executou 25 treinamentos com configurações congeladas e sementes novas. Cada treinamento teve 35 épocas. Esses 79 treinamentos são pequenos e não equivalem a testes em datasets de visão. Seus resultados completos estão em `synthetic/runs.json`.

O piloto LeNet passou por duas execuções funcionais de uma época em pixels e rótulos artificiais: AdaSym linear e AdamW nativo. Os logs em `functional/` têm `smoke: true`. Essas execuções verificam integração e formato dos logs e não medem desempenho em Fashion-MNIST. Nenhum novo treino completo de Fashion-MNIST ou CIFAR-10 foi realizado nesta revisão.

## Integridade e figuras

Os hashes das cópias dos arquivos de origem constam em `data/source_manifest.json`. Os arrays foram carregados com `allow_pickle=False`. A auditoria encontrou 70 NaNs em `loss_matrix.npy`, preservados como parte do histórico.

Os gráficos históricos e sintéticos foram gerados a partir dos arrays/logs e inspecionados visualmente. As curvas de média não foram apresentadas como intervalos de confiança. A incerteza sobre os metadados originais permanece explícita.

A apresentação dos testes do autor em `original_tests/` foi recalculada diretamente dos 16 arrays originais. Ela inclui as condições do metadata, a distinção entre índices de varredura e hiperparâmetros conhecidos, as curvas de Fashion-MNIST/CIFAR-10 e os contadores acumulados e por incremento. Essa revisão de apresentação não altera os dados nem a regra implementada.

## Limitações conhecidas

A implementação PyTorch é uma referência com diagnósticos que sincronizam o dispositivo. Não foi medida em GPU nem em precisão mista. A paridade numérica testada usa float64. Não há otimizações fused/foreach, gradientes esparsos ou modo diferenciável. O código prioriza rastreabilidade da hipótese.
