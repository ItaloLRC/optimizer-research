# AdaSym — pesquisa experimental em otimização

[English](README.md) | Português (Brasil)

Pesquisa em algoritmos de otimização para machine learning, desenvolvida por **Ítalo Capuzzo** em um grupo de estudos de machine learning e IA. O projeto combina **descoberta simbólica, experimentação com redes neurais e ciência de dados** para investigar regras de atualização adaptativas.

A análise de curvas de loss, varreduras de hiperparâmetros e estatísticas internas do otimizador orienta a identificação de padrões e a revisão das hipóteses. Esse processo acompanha a evolução da fórmula do AdaSym, desde a expressão descoberta até a incorporação de memória direcional, limitação do boost e ponderação temporal.

[Análise dos experimentos](results/research/report.md) · [Notebook de diagnóstico](notebooks/threshold-diagnostics.ipynb) · [Formulação](docs/pesquisa.md) · [Busca simbólica no Kaggle](https://www.kaggle.com/code/italocapuzzo/symbolic-optimizer-discovery-on-mnist-via-gp)

## Regra de atualização

O AdaSym adapta o tamanho do passo à consistência recente dos gradientes em cada coordenada. Uma memória direcional persistente reduz o denominador da atualização, produzindo um **boost element-wise**. O piso relativo `v/c` limita essa amplificação.

$$d=\max\left(v-s\,m_2^2,\frac{v}{c}\right),\qquad \theta_{k+1}=\theta_k-\eta\frac{m_1}{\sqrt d}.$$

| Componente | Função |
|---|---|
| `m1` | Direção da atualização, obtida da memória dos gradientes |
| `v` | Escala de normalização baseada nos gradientes ao quadrado |
| `m2` | Memória direcional usada na correção do denominador |
| `s` | Peso da correção; pode variar durante o treinamento |
| `c` | Controle do piso e da amplificação máxima |
| `eta` | Taxa de aprendizado |

Na variante **AdaSymSigmoid**, uma sigmoide reduz gradualmente a participação da correção direcional ao longo das épocas.

## Análise de dados na evolução da fórmula

Os experimentos investigam tanto o resultado do treinamento quanto o funcionamento interno do otimizador. A comparação das curvas avalia a dinâmica de convergência; os contadores de boost examinam como a regra atua sobre as coordenadas; as varreduras estudam sua resposta aos hiperparâmetros.

| Etapa da formulação | Aspecto investigado |
|---|---|
| Expressão obtida por busca simbólica | Estruturas candidatas para a atualização |
| Memória ao quadrado no denominador | Relação entre memória direcional e escala dos gradientes |
| Piso relativo `v/c` | Controle da amplificação do passo |
| Memória separada `m2` | Resposta da correção a seus próprios coeficientes |
| Peso temporal por sigmoide | Evolução da correção durante o treinamento |

Essa sequência representa um processo de investigação orientado por dados: observação de padrões, comparação de variantes e revisão da regra de atualização.

## Diagnóstico do boost e sensibilidade do threshold

Uma análise central utiliza os registros `AdaSym_fminist_c4`, `c2` e `full`, obtidos em **LeNet/Fashion-MNIST durante 150 épocas**, com piso `v/4`.

Os contadores acompanham o boost direcional produzido pela janela de consistência. `full` está associado ao boost integral limitado pelo threshold; `c4` e `c2` acompanham faixas de boost parcial relacionadas aos limites `v/4` e `v/2`, com **`c4` incluído em `c2`**.

O diagnóstico examina a evolução dos contadores acumulados, seus incrementos por época e a composição interna de `c2`. Esses dados permitem investigar como o comportamento do boost muda durante o treinamento e como as faixas monitoradas se relacionam.

![Dinâmica dos contadores de boost](results/research/threshold_dynamics.png)

[Relatório visual](results/research/report.html) · [Contadores por época](results/research/threshold_by_epoch.csv) · [Metodologia](docs/investigacao_dados.md)

## Experimentos e comparações

| Frente | Dados e análise |
|---|---|
| Comparação de otimizadores | AdaSym, AdaSymSigmoid e AdamW em Fashion-MNIST e CIFAR-10 |
| Dinâmica de treinamento | Curvas de loss, AUC normalizada e comportamento nas últimas épocas |
| Boost por coordenada | Séries acumuladas, incrementos e composição dos contadores |
| Sensibilidade aos hiperparâmetros | Learning rate, coeficientes de memória e transição da sigmoide |
| Descoberta simbólica | Evolução do fitness, tamanho das expressões e fórmulas selecionadas |

![Curvas e comparações dos experimentos](results/research/loss_diagnostics.png)

O [relatório de análise](results/research/report.md) reúne tabelas e interpretações. Os resultados são acompanhados por CSVs, figuras PNG/SVG e um manifesto com os hashes dos dados e os parâmetros da análise.

## Descoberta simbólica

O notebook **Symbolic Optimizer Discovery on MNIST via GP** explora expressões construídas a partir do gradiente e de seus momentos, usando programação genética com DEAP e PyTorch.

- Seleção por torneio, cruzamento, mutação e elitismo.
- Avaliação de candidatos em LeNet/MNIST por loss de validação relativa a AdamW.
- Controle de complexidade e profundidade das expressões.
- Acompanhamento do fitness e das fórmulas selecionadas.

[Código do notebook](notebooks/symbolic-optimizer-discovery-on-mnist-via-gp.ipynb) · [Guia de execução](notebooks/README.pt-BR.md)

## Executar

Requer Python 3.10 ou superior. Na raiz do repositório:

```bash
python -m venv .venv
# Git Bash no Windows:
source .venv/Scripts/activate
python -m pip install -e ".[plots,docs]"
python scripts/analyze_research.py
```

O comando gera o relatório, os gráficos e as tabelas em `results/research/`. O notebook `notebooks/threshold-diagnostics.ipynb` permite percorrer a análise em um ambiente Jupyter.

```bash
python -m unittest discover -s tests -v
python scripts/run_synthetic.py --epochs 35 --output results/runs/synthetic
```

Para os experimentos em PyTorch:

```bash
python -m pip install -e ".[torch]"
python scripts/run_fashion_mnist.py --download --method linear --epochs 30 --seed 101
python scripts/run_fashion_mnist.py --method adamw --epochs 30 --seed 101
```

## Organização

| Pasta | Conteúdo |
|---|---|
| `data/original/` | Registros NumPy dos experimentos |
| `notebooks/` | Descoberta simbólica e diagnóstico do boost |
| `src/optimizer_research/` | Implementações NumPy/PyTorch e funções de análise |
| `scripts/` | Análises, visualizações e experimentos |
| `results/research/` | Relatório, tabelas, figuras e proveniência |
| `docs/` | Formulação e metodologia |
| `references/` | Apresentação da pesquisa |
| `tests/` | Verificação das implementações e dos cálculos |
