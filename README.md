# AdaSym

AdaSym é um otimizador experimental para redes neurais que adapta o tamanho do passo à consistência recente dos gradientes. Quando os gradientes mantêm uma direção, o método permite atualizações maiores. Um limite no denominador controla essa amplificação, e uma variante com sigmoide reduz o incentivo ao longo do treinamento.

A pesquisa usa regressão simbólica para explorar fórmulas de atualização. As expressões promissoras passam por simplificação e testes empíricos, até restringir a busca a regras que possam ser interpretadas e comparadas com AdamW. O objetivo é reduzir o número de épocas necessário para chegar a uma loss baixa, preservando o comportamento estável no final do treino.

Os experimentos incluem Fashion-MNIST com LeNet e CIFAR-10 com ResNet-20, além de varreduras de hiperparâmetros. As curvas dessas duas comparações, descritas com a mesma taxa de aprendizado, apoiam a hipótese de convergência mais rápida com oscilação final da loss comparável à de AdamW. A seção de resultados apresenta as medidas e as exceções observadas.

A [pesquisa completa](docs/pesquisa.md) desenvolve a fundamentação e as extensões do método. Para uma leitura no navegador, abra `LEIA-ME.html`. O [mapa de experimentos](docs/mapa_experimentos.md) identifica os arquivos de cada teste, e o [protocolo de avaliação](docs/protocolo.md) descreve as próximas comparações. A [proveniência](docs/proveniencia.md) registra a origem dos dados.

## Como funciona

Durante o treinamento, o gradiente indica como alterar os pesos para reduzir a função de perda, ou loss. Como cada minibatch contém exemplos diferentes, essa indicação varia de um passo para outro. Adam usa uma média dos gradientes para definir a direção e uma média dos gradientes ao quadrado para ajustar a escala da atualização.

AdaSym acrescenta uma memória da direção, m2, à normalização. Gradientes que concordam tendem a manter essa memória forte. Mudanças de sinal tendem a produzir cancelamento. Essa é a ideia da janela de consistência: usar o histórico recente para decidir quanto ampliar o passo em cada coordenada.

### Regra de atualização

$$d=\max\left(v-s \cdot m_2^2,\frac{v}{c}\right),\qquad \theta_{k+1}=\theta_k-\eta\frac{m_1}{\sqrt d}.$$

| Termo | Significado |
|---|---|
| m1 | Média dos gradientes que define a direção da atualização |
| v | Média dos gradientes ao quadrado, usada para normalizar o passo |
| m2 | Memória adicional da direção recente; tem um papel distinto de v |
| s | Peso da correção de consistência, constante ou variável no tempo |
| c | Divisor do piso v/c, que limita o aumento do passo |
| eta | Taxa de aprendizado |
| theta | Pesos da rede |

A correção s·m2² diminui d. Como a atualização divide por raiz(d), essa diminuição aumenta o passo. O `max` impede que d caia abaixo de v/c. Para o mesmo estado dos momentos, a amplificação adaptativa fica entre 1 e raiz(c): com c=4, até 2×; com c=8, até aproximadamente 2,83×.

Por exemplo, considere uma coordenada com v=4, m1=0,2, c=4 e s=1:

| m2² | d | Tamanho do passo, sem o fator eta |
|---:|---:|---:|
| 0 | 4 | 0,1000 |
| 1 | 3 | 0,1155 |
| 3 ou mais | 1, limitado pelo piso | 0,2000 |

No treinamento, m1, m2 e v mudam juntos. O limite descreve quanto o denominador pode ampliar o passo em um dado estado; ele não garante descida da loss a cada atualização. Epsilon, correção de viés e weight decay completam a especificação numérica da implementação.

### Redução temporal da correção

A variante AdaSym sem redução temporal mantém s=1. AdaSymSigmoid usa uma sigmoide decrescente, com época de transição T e época atual e:

$$s(e)=\sigma(T-e)=\frac{1}{1+\exp(e-T)}.$$

Antes de T, o método aplica quase toda a correção de consistência. Em T, o peso é 0,5. Depois, o denominador se aproxima de v. Essa transição procura combinar passos maiores no início com uma normalização mais próxima da usada por Adam no final.

Uma configuração registrada em `metadata.txt` usa `sigmoid(15 − cur_epoch)` e `max(v_hat − alpha·s_hat², v_hat/8)`. Nessa implementação, `alpha` é o peso temporal s, e `s_hat` é a memória m2. Os testes incluem pisos v/4 e v/8. A recorrência original de m2, incluindo o papel de beta3 e beta4, ainda precisa ser recuperada para reproduzir esses treinamentos exatamente.

## Resultados disponíveis

Os dados dos experimentos estão em `data/original/`. O [relatório dos experimentos](results/original_tests/report.md) contém os números completos e as condições descritas no metadata. As losses não identificam se o registro pertence ao treino, à validação ou ao teste; por isso, as comparações abaixo descrevem a dinâmica das curvas, sem concluir sobre acurácia ou generalização.

![Curvas de treinamento em Fashion-MNIST e CIFAR-10](results/original_tests/author_training.png)

### Rapidez de convergência e comportamento final

Para medir a redução da loss em número de épocas, foi usado o primeiro trecho de cinco épocas consecutivas em que a curva média das três trajetórias permanece em loss ≤ 0,1. A oscilação final é a média dos desvios padrão temporais de cada trajetória nas últimas 20 épocas.

| Experimento | Método | Época até o alvo | Loss média final, 20 épocas | DP temporal final, 20 épocas |
|---|---|---:|---:|---:|
| Fashion-MNIST / LeNet | AdamW | 73 | 0,043519 | 0,003417 |
| Fashion-MNIST / LeNet | AdaSym | 63 | 0,039634 | 0,003163 |
| CIFAR-10 / ResNet-20 | AdamW | 22 | 0,018017 | 0,004495 |
| CIFAR-10 / ResNet-20 | AdaSymSigmoid | 16 | 0,015440 | 0,004120 |

Nos pares descritos com LR 1e-3, AdaSym chega ao alvo antes de AdamW e termina com loss média menor. A variabilidade temporal no trecho final também é ligeiramente menor nessas duas comparações.

**A hipótese apoiada por esses resultados é que AdaSym pode acelerar a convergência em número de épocas, mantendo estabilidade final comparável à de AdamW nas configurações avaliadas.** Aqui, estabilidade se refere à oscilação observada da loss. Demonstrar igualdade de estabilidade exige uma comparação estatística própria.

O alvo, as cinco épocas consecutivas e a janela de 20 épocas são resumos exploratórios escolhidos após observar as curvas. O DP temporal inclui qualquer tendência residual no trecho final. Esses números não medem tempo de execução nem dispersão entre sementes. O [cálculo de convergência](results/original_tests/convergence.md) apresenta a definição das medidas e os resultados por arquivo.

Há dois controles que limitam a conclusão. AdamW com LR 2× alcança o mesmo alvo na época 54, antes de AdaSym. Já a variante AdaSym sem redução temporal alcança o alvo na época 60, mas apresenta oscilação final maior, com DP temporal de 0,007535. O efeito depende da taxa de aprendizado e da variante usada.

### Comparações de loss

As médias nas últimas dez épocas e sobre a curva inteira complementam a análise de convergência:

| Experimento | Método | Média nas últimas 10 épocas | Média na curva inteira |
|---|---|---:|---:|
| Fashion-MNIST, 100 épocas | AdaSym | 0,290194 | 0,291404 |
| Fashion-MNIST, 100 épocas | AdaSymSigmoid | 0,282255 | 0,287134 |
| Fashion-MNIST, 100 épocas | AdamW | 0,289600 | 0,302175 |
| LeNet, 150 épocas | AdaSym | 0,038349 | 0,120462 |
| LeNet, 150 épocas | AdamW | 0,041992 | 0,135194 |
| LeNet, 150 épocas | AdaSym sem redução temporal | 0,038083 | 0,117264 |
| LeNet, 150 épocas | AdamW com LR 2× | 0,041124 | 0,109575 |
| CIFAR-10, 150 épocas | AdaSymSigmoid | 0,015382 | 0,069594 |
| CIFAR-10, 150 épocas | AdamW | 0,017411 | 0,081801 |

A rodada de 100 épocas compara AdaSym, AdaSymSigmoid e AdamW em Fashion-MNIST. O grupo LeNet de 150 épocas usa LR 1e-3; seu registro cita piso v/4 e valores (0,75; 0,35), com uma ambiguidade na associação entre arquivos e redução temporal. A rodada CIFAR-10 usa ResNet-20, LR 1e-3, weight decay 1e-2, transição na época 15 e piso v/8.

O controle LR 2× tem dois rótulos de dataset nos registros: MNIST no metadata e Fashion-MNIST na [apresentação original](references/AdaSym-original.pptx). A [auditoria dos arquivos](results/historical/report.md) detalha essa diferença, a identificação das trajetórias e as demais lacunas de reprodução.

### Varreduras de hiperparâmetros

| Teste | Condições | Resultado |
|---|---|---|
| LR de AdaSym, 80 pontos | 300 épocas, 10% de Fashion-MNIST, AUC de loss, WD 1e-2, valores (0,75; 0,35) | Mínimo 101,313248 no índice 54. O eixo de LRs não foi salvo |
| LRs comparativas, 27 pontos | Grade de 1e-4 a 9e-2 reconstruída do metadata | Mínimos 109,020409 e 105,941498; LRs inferidas de 0,02 e 0,04 |
| Transição da sigmoide | LeNet, 20% de Fashion-MNIST, 150 épocas, LR 1e-3, WD 1e-2, faixa de 0 a 50 | Menor AUC 121,601784 no índice 17. A época exata depende do eixo original |
| Coeficientes de m2 | Duas grades beta3 × beta4 de 10 × 70 | Mínimos em (0,584341; 0,740568) e (0,674355; 0,406159) |
| Ativação do piso v/4 | LeNet, Fashion-MNIST, 150 registros | Razões acumuladas finais de 9,9777% para c4/full e 14,0758% para c2/full |
| Divisor do piso | LeNet, 20% de Fashion-MNIST, 150 épocas, LR 1e-3, WD 1e-2, sem decaimento, divisores de 2 a 20 | Teste registrado como `div_aucloss`; o arquivo numérico ainda não está no repositório |

As varreduras investigam a duração da janela, a intensidade da correção e a frequência com que o limite do passo atua. Seus mínimos orientam novas configurações. Os registros ainda precisam de identificação completa das sementes, dos splits e das recorrências para permitir uma confirmação independente.

## Implementações e extensões

O repositório contém uma variante de janela comum: a média dos gradientes e a média dos gradientes ao quadrado usam a mesma ponderação temporal. Uma razão normalizada q mede a consistência, e uma função h(q) controla a correção do denominador. Essa extensão tem uma definição executável; a recorrência histórica de m2 permanece documentada como uma lacuna.

- O [ensaio sintético](results/synthetic/report.md) compara funções h(q), usa cinco inicializações na confirmação e inclui AdamW com LR maior. Nessa tarefa, o controle com LR maior superou as variantes AdaSym.
- A [verificação da implementação](results/verification.md) cobre recorrência, limites de ganho, gradientes, paridade NumPy/PyTorch e retomada de estado.

Esses resultados delimitam a hipótese de aceleração. Sua confirmação requer recuperar a implementação original, repetir as comparações com orçamento equivalente de ajuste e avaliar a fórmula em tarefas que não participaram da seleção.


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

O exemplo usa a variante de janela comum. A opção `rule="historical"` implementa o denominador AdaSym com uma média exponencial explicitada. A equivalência com os treinamentos históricos depende de recuperar a recorrência original de m2.

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

O piloto em `scripts/run_fashion_mnist.py` usa LeNet e baixa Fashion-MNIST com `--download`. Sua integração passou por uma verificação funcional com dados artificiais. Os resultados de visão apresentados acima pertencem aos treinamentos históricos; o piloto não acrescenta uma nova rodada completa em Fashion-MNIST ou CIFAR-10.

```bash
# Piloto com treino e validação, sem consultar o teste oficial:
python scripts/run_fashion_mnist.py --download --method linear --epochs 30 --seed 101
python scripts/run_fashion_mnist.py --method adamw --epochs 30 --seed 101
python scripts/run_fashion_mnist.py --method adamw_scaled --epochs 30 --seed 101
# Usar --evaluate-test apenas depois de congelar fórmula e hiperparâmetros.
```

O gerador de `LEIA-ME.html` requer `python -m pip install -e ".[docs]"` e pode ser executado com `python scripts/build_reading_copy.py`. O arquivo funciona offline.

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

O workflow `.github/workflows/tests.yml` executa os testes no GitHub Actions. O projeto ainda não declara uma licença de distribuição.
