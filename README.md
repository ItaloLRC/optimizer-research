# Optimizer Research · AdaSym

Pesquisa sobre **otimizadores com uma janela de consistência dos gradientes**, usando busca simbólica para explorar e restringir as regras de atualização.

A ideia central é amplificar o passo quando direção e magnitude do gradiente persistem. A pesquisa começou com a exploração de fórmulas por regressão simbólica, seguida de ajustes e testes empíricos. A apresentação AdaSym e os experimentos do autor em `Data`, descritos no `metadata.txt`, constituem a base deste projeto.

**Comece pela [pesquisa completa](docs/pesquisa.md)** ou abra `LEIA-ME.html` no navegador. Consulte também o [mapa dos testes recebidos](docs/mapa_experimentos.md), o [protocolo de experimentos](docs/protocolo.md), a [proveniência](docs/proveniencia.md) e os resultados abaixo.

## Formulação

### A regra do AdaSym apresentada nos slides

O ponto de partida é uma atualização do tipo Adam: usar a direção acumulada dos gradientes, m1, e dividir seu tamanho pela raiz de v, uma memória da magnitude dos gradientes ao quadrado. AdaSym altera esse denominador para permitir passos maiores quando a memória da direção, m2, permanece forte.

Na notação dos slides, a regra final é:

$$d=\max\left(v-s\,m_2^2,\frac{v}{c}\right),\qquad \theta_{k+1}=\theta_k-\eta\frac{m_1}{\sqrt d}.$$

| Termo | Papel na atualização |
|---|---|
| m1 | Memória da direção usada para mover os pesos |
| v | Memória da magnitude dos gradientes ao quadrado; fornece a escala de normalização |
| m2 | Memória adicional da direção, associada à janela de consistência; **não é o segundo momento v** |
| s | Peso da correção de consistência: 1 na variante constante ou uma sigmoide na variante temporal |
| c | Divisor do piso do denominador; os testes incluem v/4 e v/8 |
| eta | Taxa de aprendizado |

**Como funciona:** quando os gradientes recentes concordam, sua memória m2 tende a manter magnitude. Subtrair s·m2² reduz d e aumenta o passo. Quando há cancelamento das direções, a correção tende a diminuir. O `max` impede que d fique abaixo de v/c e limita a amplificação relativa a sqrt(c), para o mesmo estado dos momentos. Com c=4, o limite é 2×; com c=8, aproximadamente 2,83×. Esse é um limite relativo do passo adaptativo, não uma garantia de convergência.

Na variante **AdaSymSigmoid**, os slides usam uma sigmoide decrescente. Escrevendo T para a época de transição e e para a época atual:

$$s(e)=\sigma(T-e)=\frac{1}{1+\exp(e-T)}.$$

Antes de T, a correção permanece forte. Em T, seu peso é 0,5. Depois, ela diminui e o denominador se aproxima de v. O metadata contém `sigmoid(15 − cur_epoch)` e `max(v_hat − alpha·s_hat², v_hat/8)`: nesse trecho, `alpha` corresponde ao peso s dos slides e `s_hat` corresponde à memória m2. A recorrência exata de m2, incluindo beta3/beta4, não foi fornecida e não é substituída por uma definição inventada.

Exemplo com v=4, m1=0,2, c=4 e s=1:

| Valor de m2² | Denominador d | Tamanho do passo, sem o fator eta |
|---:|---:|---:|
| 0 | 4 | 0,1000 |
| 1 | 3 | 0,1155 |
| 3 ou mais | 1, limitado pelo piso | 0,2000 |

Essa é a explicação do otimizador dos slides. A [pesquisa completa](docs/pesquisa.md) também acompanha a sequência de modificações da fórmula descoberta até essa regra. Epsilon, correção de viés e weight decay exigem uma especificação de implementação além da equação resumida da apresentação.

## Resultados disponíveis

### Testes realizados pelo autor

Os resultados abaixo vêm dos seus arquivos originais e das condições do metadata. O [relatório dos testes do autor](results/original_tests/report.md) reúne tabelas, gráficos, arquivos associados e a interpretação de cada experimento. Os valores das tabelas são losses registradas; o split não está identificado nos arquivos recebidos.

| Teste | Condições registradas | Resultado disponível |
|---|---|---|
| AdaSym × AdaSymSigmoid × AdamW | Fashion-MNIST, 100 épocas | Média da loss na curva: **0,291404 / 0,287134 / 0,302175**, respectivamente. Sigmoid tem a menor média também nas últimas 10 épocas |
| AdaSym × AdamW e variante sem redução temporal | LeNet, Fashion-MNIST, 150 épocas, LR 1e-3; o metadata cita piso v/4 e valores (0,75; 0,35) em um dos trechos | Média das últimas 10 épocas: **AdaSym 0,038349**, **AdamW 0,041992** e **AdaSym nored 0,038083**. A associação exata de cada variante ao scheduler está indicada no relatório |
| AdaSymSigmoid × AdamW | CIFAR-10, ResNet-20, 150 épocas, LR 1e-3, WD 1e-2, transição na época 15, piso v/8 | Média nas últimas 10 épocas: **0,015382 × 0,017411**. Média na curva: **0,069594 × 0,081801** |
| AdaSym × AdamW com LR 2× | LeNet, 150 épocas; metadata diz MNIST e slides dizem Fashion-MNIST | AdamW 2× tem menor média na curva (**0,109575 × 0,120462**), enquanto AdaSym tem menor média nas últimas 10 épocas (**0,038349 × 0,041124**) |
| Varreduras de LR | AdaSym: 300 épocas, 10% de Fashion-MNIST, AUC, WD 1e-2, valores (0,75; 0,35); mais dois vetores de 27 LRs | Na varredura de 80 pontos, mínimo **101,313248**, índice 54, sem eixo de LR salvo. Nos vetores de 27 pontos, mínimos **109,020409** e **105,941498**; LRs inferidas do metadata **0,02** e **0,04** |
| Transição da sigmoide | LeNet, 20% de Fashion-MNIST, 150 épocas, LR 1e-3, WD 1e-2, faixa descrita 0–50 | Menor AUC salva **121,601784**, índice 17 da grade. O relatório distingue o índice da época exata, cujo eixo não foi salvo |
| Coeficientes de m2 | Duas grades beta3 × beta4 de 10 × 70 posições | Mínimos em **(0,584341; 0,740568)** e **(0,674355; 0,406159)**, correspondentes aos destaques dos slides |
| Ativação do piso | LeNet, Fashion-MNIST, piso v/4; vetores com 150 registros | Razões acumuladas finais: **9,9777%** para c4/full e **14,0758%** para c2/full |
| Divisor do piso | LeNet, 20% de Fashion-MNIST, 150 épocas, LR 1e-3, WD 1e-2, sem decaimento, divisores 2–20 | Experimento descrito no metadata como `div_aucloss`; arquivo numérico não incluído no material recebido |

![Testes realizados pelo autor: Fashion-MNIST, LeNet e CIFAR-10](results/original_tests/author_training.png)

As comparações mostram comportamentos distintos no início e no final do treino. A identificação de seeds e do split permite aprofundar as conclusões; a [auditoria dos arquivos](results/historical/report.md) conserva essas observações técnicas e as diferenças entre metadata e slides.

### Implementação e experimentos adicionais desta revisão

A revisão acrescentou uma **variante de janela comum**, em que uma razão normalizada q mede consistência e uma função h(q) controla o denominador. Essa é uma proposta complementar; ela não substitui a fórmula original nem define retroativamente o m2 dos seus testes.

- [Teste adicional em MLP sintético](results/synthetic/report.md): triagem de funções h(q), cinco inicializações de confirmação e controle de LR maior. Esse ensaio não demonstrou vantagem sobre AdamW com LR maior e não constitui uma reavaliação dos seus testes de visão.
- [Verificação da implementação adicional](results/verification.md): recorrência, limites, gradientes, paridade NumPy/PyTorch e checkpoint.

## Executar

Python 3.10 ou superior. No terminal, dentro desta pasta:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -e ".[plots]"
python -m unittest discover -s tests -v
python scripts/analyze_original.py
python scripts/summarize_author_experiments.py
python scripts/run_synthetic.py --epochs 35 --output results/runs/synthetic
```

O núcleo e o MLP sintético dependem apenas de NumPy. O gerador de gráficos lê a execução de referência em `results/synthetic`. Para refazer os gráficos dessa execução:

```bash
python scripts/run_synthetic.py --epochs 35 --output results/synthetic
python scripts/plot_results.py
```

### PyTorch

O exemplo a seguir usa a **variante adicional de janela comum**. A opção `rule="historical"` do código reconstrói o formato do denominador dos slides com uma EMA explicitada, mas não recupera a recorrência de m2 que falta no material original.

```bash
python -m pip install -e ".[torch]"
python -m unittest discover -s tests -v
```

```python
from optimizer_research import Config
from optimizer_research.torch_optimizer import AdaSymTorch

optimizer = AdaSymTorch(model.parameters(), Config(
    lr=1e-3, beta_window=0.9, cap=4,
    alpha=1.0, warmup_steps=20, rule="linear",
))
for inputs, targets in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()
```

O modo sigmoide requer `optimizer.set_progress(step / max(total_steps - 1, 1))` antes de cada passo. Gradientes ausentes pulam atualização e weight decay. Gradientes esparsos ou complexos não são suportados. A implementação mantém quatro estados de momento e coleta diagnósticos que podem sincronizar a GPU. É uma referência de pesquisa, sem otimizações de desempenho.

Para um teste piloto em Fashion-MNIST, consulte `scripts/run_fashion_mnist.py`. Esse script baixa o dataset somente com `--download`. Ele é uma nova implementação de LeNet, não uma recuperação do modelo histórico. A entrega inclui sua verificação funcional com dados artificiais, mas não resultados novos de Fashion-MNIST/CIFAR-10.

```bash
# Piloto com treino e validação, sem consultar o teste oficial:
python scripts/run_fashion_mnist.py --download --method linear --epochs 30 --seed 101
python scripts/run_fashion_mnist.py --method adamw --epochs 30 --seed 101
python scripts/run_fashion_mnist.py --method adamw_scaled --epochs 30 --seed 101
# Usar --evaluate-test apenas depois de congelar fórmula e hiperparâmetros.
```

O gerador de `LEIA-ME.html` requer `python -m pip install -e ".[docs]"` e pode ser executado com `python scripts/build_reading_copy.py`. A cópia entregue já está pronta e funciona offline.

## Organização

| Pasta | Conteúdo |
|---|---|
| `docs/` | Fundamentação, protocolo e proveniência |
| `src/optimizer_research/` | Otimizadores, gramática e MLP sintético |
| `scripts/` | Auditoria, triagem, gráficos e piloto em visão |
| `tests/` | Testes da regra e da implementação |
| `data/original/` | Fontes NumPy e metadados preservados |
| `references/` | Apresentação original |
| `results/` | Relatórios, logs e figuras reproduzíveis |

O arquivo `.github/workflows/tests.yml` deixa os testes preparados para uma publicação futura. Esta entrega é local e não foi enviada ao GitHub. A autoria e a licença de distribuição devem ser definidas antes de uma publicação aberta.
