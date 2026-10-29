# AdaSym: descoberta simbólica de otimizadores com janela de consistência

**Versão de pesquisa: 29 de outubro de 2026.** A base desta pesquisa é o AdaSym apresentado nos slides e testado pelo autor nos experimentos descritos no metadata. O texto explica primeiro essa formulação e seus resultados. A janela comum normalizada e os experimentos sintéticos aparecem depois como extensões adicionais desta revisão.

## 1. Pergunta de pesquisa

Uma rede neural pode convergir mais rapidamente quando o otimizador aumenta seletivamente seus passos nas coordenadas cujos gradientes mantêm direção e magnitude durante uma janela recente?

A hipótese do projeto é que a persistência do gradiente informa o quanto se pode amplificar o passo. A regressão simbólica serve para explorar regras candidatas, identificar estruturas promissoras e restringir progressivamente o espaço de busca. A avaliação empírica deve separar um ganho causado pela consistência de um ganho obtido apenas por uma taxa de aprendizado maior.

O objetivo verificável é melhorar a eficiência até uma loss ou acurácia predefinida, mantendo qualidade de teste e orçamento de ajuste comparáveis. Convergência em menos épocas, tempo até o alvo, custo de estado e generalização são resultados diferentes e devem aparecer separadamente.

## 2. AdaSym na apresentação: regra e interpretação

### A fórmula final dos slides

AdaSym mantém uma direção de atualização m1 e modifica a escala usada para normalizar o passo. A memória adicional m2 representa a persistência recente do gradiente. A ideia é dar um passo maior quando essa memória permanece forte em relação à escala v dos gradientes ao quadrado.

Na notação da apresentação:

$$d=\max\left(v-s\,m_2^2,\frac{v}{c}\right).$$

$$\theta_{k+1}=\theta_k-\eta\frac{m_1}{\sqrt d}.$$

| Símbolo | Significado e efeito |
|---|---|
| g | Gradiente calculado no passo atual |
| m1 | Memória da direção do gradiente que determina para onde os pesos se movem |
| v | Memória da magnitude dos gradientes ao quadrado, usada como escala de normalização |
| m2 | Memória adicional da direção do gradiente, relacionada à janela de consistência |
| s | Peso dado à correção m2²: constante na variante sem redução temporal, variável na sigmoide |
| c | Divisor do piso v/c. Valores 4 e 8 aparecem nos testes do autor |
| eta | Taxa de aprendizado |
| theta | Pesos da rede neural |

**m2 e v têm papéis diferentes.** O nome m2 não significa que ele seja o segundo momento do Adam. A expressão v já cumpre o papel da escala de segundo momento. A recorrência de m2, objeto das grades beta3/beta4, não está incluída nos arquivos recebidos. Sua interpretação como memória da consistência decorre da fórmula e da hipótese apresentada pelo autor, sem fixar coeficientes que não foram documentados.

### A janela de consistência dentro da fórmula

Quando gradientes recentes apontam de maneira persistente na mesma direção, uma memória ponderada desses gradientes tende a manter magnitude. Quando os sinais se cancelam, essa memória tende a diminuir. Na regra dos slides, m2² mede a intensidade dessa memória e s controla quanto ela participa da atualização.

O mecanismo é direto: subtrair s·m2² reduz d. Como o passo divide por raiz(d), um denominador menor permite um passo maior. O `max` interrompe essa redução em v/c. Isso evita normalização por um valor negativo ou excessivamente pequeno em relação a v. Se v=0, a fórmula resumida ainda exige estabilização numérica na implementação.

Exemplo para uma coordenada, mantendo v=4, m1=0,2, c=4 e s=1:

| Memória m2² | v − s·m2² | Piso v/c | d escolhido | Tamanho do passo / eta |
|---:|---:|---:|---:|---:|
| 0 | 4 | 1 | 4 | 0,1000 |
| 1 | 3 | 1 | 3 | 0,1155 |
| 3 | 1 | 1 | 1 | 0,2000 |
| 4 | 0 | 1 | 1 | 0,2000 |

O exemplo isola o efeito da memória no denominador. Em um treinamento, m1, m2 e v mudam juntos. Uma consistência maior permite uma amplificação relativa maior; ela não garante que o passo absoluto aumente em toda comparação entre épocas.

Para s ≥ 0 e c ≥ 1, o piso garante `v/c ≤ d ≤ v`. Sem epsilon, e para v>0, o ganho sobre o denominador raiz(v) fica entre 1 e raiz(c), considerando os mesmos momentos naquele instante. Assim, c=4 permite até 2× e c=8 até aproximadamente 2,83×. O limite descreve a parte adaptativa do passo e não prova convergência.

### O papel da sigmoide

Na versão AdaSym sem redução temporal, a correção pode permanecer com s=1. Na versão AdaSymSigmoid, o peso da correção diminui com a época. Escrevendo T para o ponto de transição e e para a época atual, a fórmula do slide 7 fica:

$$s(e)=\sigma(T-e)=\frac{1}{1+\exp(e-T)}.$$

| Época em relação a T | Peso aproximado s | Efeito |
|---|---:|---|
| T − 5 | 0,9933 | Correção de consistência quase integral |
| T | 0,5 | Metade do peso da correção |
| T + 5 | 0,0067 | Denominador próximo da escala v |

O metadata contém o trecho:

```python
alpha = torch.sigmoid(torch.tensor(15 - cur_epoch))
val = torch.clamp(v_hat - s_hat.square() * alpha, min=v_hat/8)
```

Nesse trecho, `alpha` é o peso s dos slides, `s_hat` é a memória m2, `v_hat` é a escala v e o divisor c é 8. O código indica como combinar essas quantidades, mas não inclui suas recorrências. O gráfico ilustrativo da apresentação usa T=5 e o metadata cita T=15 em uma configuração. São escolhas do ponto de transição, não definições diferentes da sigmoide.

Quando s tende a zero, d tende a v e a correção de consistência desaparece. A atualização só coincide com Adam/AdamW completo se os momentos, bias correction, epsilon e weight decay também coincidirem. A equação resumida dos slides não fixa todos esses detalhes.

### Como a fórmula evoluiu na pesquisa

Os slides documentam as seguintes mudanças:

1. **Fórmula encontrada:** `m1 / sqrt(abs(v − m1))`, mostrada no slide 4.
2. **Quadrado da memória:** `m1 / sqrt(abs(v − m1²))`, para compatibilizar a ordem da memória com a escala v.
3. **Piso relativo:** `m1 / sqrt(max(v − m1², v/c))`, limitando a amplificação do passo.
4. **Memória separada:** `m1 / sqrt(max(v − m2², v/c))`, permitindo investigar coeficientes próprios para a memória usada no denominador.
5. **Peso temporal:** `m1 / sqrt(max(v − s·m2², v/c))`, com a sigmoide controlando por quanto tempo a correção permanece forte.

A primeira expressão subtrai grandezas de ordens diferentes se v representa E[g²] e m1 representa E[g]. A versão com m1² corrige a compatibilidade dimensional. O valor absoluto transforma qualquer diferença negativa em uma normalização positiva, mas sua interpretação depende da diferença entre as janelas. O piso v/c torna explícito um limite relativo do denominador. A sigmoide controla quanto da correção permanece ao longo do treinamento.

As matrizes existentes permitem descrever a loss registrada e localizar valores inválidos. Elas não identificam integralmente sementes, split dos dados, precisão, inicializações, scheduler ou recorrência de m2. Por isso, não demonstram uma nova garantia de convergência, superioridade estatística nem reprodução da descoberta simbólica. O relatório em `results/historical/report.md` quantifica o que pode ser medido e explicita as ambiguidades.

O histórico já contém um controle relevante de escala do passo: nos arquivos de 150 posições, a média registrada é 0,135194 para AdamW, 0,120462 para AdaSym e 0,109575 para o arquivo descrito como AdamW com LR 2×. Nas últimas dez posições, AdaSym tem média menor, 0,038349 contra 0,041124 do controle 2×. Essa diferença entre velocidade inicial e loss tardia ajuda a formular a próxima comparação. O conflito de rótulo MNIST/Fashion-MNIST e a ausência do código impedem tratar esses números como um experimento controlado já estabelecido. O mapa em `docs/mapa_experimentos.md` organiza as condições relatadas, incluindo os testes de LR, sigmoide, piso e betas.

## 3. Extensão opcional: janela comum com consistência normalizada

**Esta seção introduz uma proposta adicional da revisão.** As definições mu_t, u_t e q_t a seguir não constam da formulação original dos slides e não são atribuídas aos treinamentos do autor. A letra mu identifica a média do gradiente nesta extensão, enquanto s(e) continua identificando o peso temporal dos slides. A extensão usa uma janela comum para tornar a consistência normalizada interpretável e testável.

Para cada coordenada i, mantemos uma média exponencial do gradiente e uma média do gradiente ao quadrado, ambas com o mesmo beta:

$$\mu_t=\beta_w \mu_{t-1}+(1-\beta_w)g_t,$$
$$u_t=\beta_w u_{t-1}+(1-\beta_w)g_t^2.$$

Com estado inicial zero, usamos a mesma correção de viés nas duas:

$$\widehat \mu_t=\frac{\mu_t}{1-\beta_w^t},\qquad \widehat u_t=\frac{u_t}{1-\beta_w^t}.$$

A consistência proposta é

$$q_t=\operatorname{clip}\left(\frac{\widehat \mu_t^2}{\widehat u_t+\epsilon^2},0,1\right).$$

As operações são coordenada a coordenada. A expressão tem interpretação de momento normalizado: a média mantém seu peso quando os gradientes concordam e sofre cancelamento quando mudam de sinal. Oscilações de magnitude também reduzem q.

Com pesos normalizados não negativos, Jensen garante `(média g)² ≤ média(g²)`. Assim, a razão pertence a [0,1], salvo arredondamento. Usar a mesma janela e a mesma correção de viés é essencial para essa interpretação. Na subtração dos slides `v − m2²`, essa propriedade não pode ser assumida quando v e m2 usam janelas ou ponderações diferentes. A recorrência original precisa ser recuperada para estabelecer a relação exata entre eles.

### Exemplos sem ambiguidade

| Gradientes em uma janela com pesos iguais | Média | Segundo momento | Consistência sem epsilon |
|---|---:|---:|---:|
| 1, 1, 1, 1 | 1 | 1 | 1 |
| 1, −1, 1, −1 | 0 | 1 | 0 |
| 1, 3, 1, 3 | 2 | 5 | 0,8 |
| 0, 0, 0, 0 | 0 | 0 | 0 pela convenção estabilizada |

Na EMA, o caso alternante geralmente tem q pequeno e diferente de zero porque o último gradiente pesa mais. Sob um modelo estacionário com média μ e variância σ², a razão populacional é `μ²/(μ²+σ²)`. A razão das estimativas em uma janela finita não é um estimador sem viés dessa quantidade.

Esta definição combina consistência de sinal e de magnitude. Ela não é uma medida exclusiva do ângulo do gradiente global, nem um intervalo de confiança. Um gradiente persistentemente errado, provocado por viés no minibatch ou mudança na tarefa, pode ter consistência alta.

### O tamanho da janela

Uma EMA não possui corte rígido. Seu tamanho efetivo assintótico é

$$N_{\mathrm{eff}}=\frac{1+\beta_w}{1-\beta_w}.$$

Depois de t observações, o tamanho efetivo é

$$N_{\mathrm{eff},t}=\frac{1+\beta_w}{1-\beta_w}\frac{1-\beta_w^t}{1+\beta_w^t}.$$

| beta da janela | Tamanho efetivo assintótico |
|---:|---:|
| 0,35 | 2,08 |
| 0,75 | 7 |
| 0,90 | 19 |
| 0,95 | 39 |
| 0,99 | 199 |

Janelas curtas reagem cedo, mas podem confundir ruído com persistência. Janelas longas reduzem esse ruído e atrasam a resposta a inversões. No primeiro passo não nulo, q é praticamente 1 mesmo sem evidência de persistência. O warmup reduz gradualmente a amplificação inicial. Ele não elimina esse viés nem substitui a análise da janela.

## 4. Regra adicional para a janela comum

Mantemos os momentos do Adam separados da janela:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.$$

Aplicamos bias correction a m e v. A família restrita é

$$d_t=\widehat v_t\max\{1-a_t h(q_t),1/c\},$$
$$\theta_t=(1-\eta_t\lambda)\theta_{t-1}-\eta_t\frac{\widehat m_t}{\sqrt{d_t}+\epsilon}.$$

Aqui, c ≥ 1, a está em [0,1], e h leva [0,1] a [0,1], de forma crescente. A primeira hipótese é `h(q)=q`. O weight decay é desacoplado, como em AdamW. Epsilon entra fora da raiz na atualização, e epsilon² estabiliza a razão de consistência.

O sinal da atualização adaptativa vem de m. A consistência altera sua magnitude relativa. Quanto maior q, menor o denominador e maior o passo, até o limite definido por c. Esse aumento se refere ao passo que AdamW daria com os mesmos momentos naquele instante. Não implica que todo passo absoluto cresça quando q cresce, porque m e v também mudam.

### Um limite útil e seu alcance

Como `v/c ≤ d ≤ v`, a amplificação relativa ao AdamW com o mesmo estado satisfaz

$$1\leq\frac{\sqrt{\widehat v}+\epsilon}{\sqrt d+\epsilon}\leq\sqrt c.$$

Com c=4, o ganho adaptativo máximo é 2. Com c=8, é aproximadamente 2,83. Esse limite vale para a parte adaptativa, não para a atualização total que inclui weight decay. Ele não garante descida a cada passo ou convergência. Em trajetórias diferentes, os momentos dos dois otimizadores deixam de coincidir, portanto o limite não compara diretamente seus parâmetros após vários passos.

Se a=0 ou c=1, a regra coincide com AdamW para os mesmos hiperparâmetros. Se beta_window=beta2, o estado u pode compartilhar v em uma implementação otimizada. A implementação de referência guarda ambos explicitamente para facilitar inspeção. São quatro estados de momento por parâmetro, contra dois no AdamW. Não há alegação de economia de memória ou tempo.

### Consistência por coordenada ou por bloco

A implementação usa q por coordenada. Uma alternativa para uma camada é `q_bloco = ||E[g]||²/(E[||g||²]+eps²)`. Sem epsilon, essa quantidade equivale à média de q por coordenada ponderada por seu segundo momento. Ela reduz custo de diagnóstico e mistura coordenadas com comportamentos diferentes. A escolha entre as duas é uma ablação futura. Alinhamento angular puro, como cosseno entre gradientes consecutivos, responde a outra pergunta e deve ser testado separadamente.

### Sigmoide opcional

O projeto histórico reduz a correção por época. A especificação nova usa progresso normalizado p em [0,1]:

$$a_t=\alpha\min(1,t/T_w)\,\sigma((p_0-p_t)/\tau).$$

O modo constante remove apenas a sigmoide e mantém o warmup. Uma janela já adapta o ganho aos gradientes. Assim, o decaimento temporal é uma hipótese adicional, não requisito da ideia central. Comparar constante e sigmoide permite verificar se a eventual melhora depende de uma política de LR embutida. A sigmoide histórica em épocas e esta versão normalizada não são equivalentes por padrão.

## 5. Regressão simbólica e restrição do espaço

Se a busca otimiza uma fórmula diretamente pelo desempenho de treinamento, ela é busca simbólica de programas ou de regras de atualização. O termo regressão simbólica se aplica estritamente quando há pares entrada/alvo para ajustar uma expressão. Ambos podem entrar no projeto, mas o fitness e a origem dos alvos precisam ser registrados.

A proposta separa duas etapas:

1. **Exploração ampla futura:** árvores tipadas com gradientes, momentos e funções protegidas. Registrar indivíduos, pais, mutações, sementes, custo, fitness e fórmulas descartadas. Conservar soluções na fronteira de desempenho, complexidade e estabilidade. Essa etapa histórica não pode ser reproduzida com os arquivos entregues.
2. **Busca restrita executável agora:** fixar direção e momentos, e buscar somente h(q). Isso reduz o espaço a uma variável sem dimensão com domínio conhecido. O código enumera uma gramática pequena em vez de executar um algoritmo evolutivo completo.

A gramática atual contém `q`, `q*q`, `sqrt(q)`, `2*q/(1+q)` e `tanh(q)/tanh(1)`. Todas são monótonas, não negativas e limitadas em [0,1]. O piso externo limita o ganho independentemente da transformação. A normalização impede misturar dimensões incompatíveis e preserva a simetria de sinal. A invariância a reescala é aproximada na presença de epsilon.

Cada expressão usa a mesma grade de LR e as mesmas sementes de triagem. A seleção ordena pela média da loss de validação ao longo das épocas, com complexidade como desempate. O tamanho da árvore é uma aproximação de complexidade, não de latência. A busca é pequena e auditável, mas não identifica o ótimo do espaço de otimizadores nem reproduz o mecanismo original.

Uma extensão deve usar árvores tipadas de profundidade limitada, deduplicação algébrica e comparação numérica de expressões equivalentes. Testes com gradiente zero, sinal invertido, spikes e escalas diferentes devem filtrar fórmulas antes de treinar redes. Operações protegidas precisam registrar quando a proteção atua, pois ela altera a fórmula efetiva.

## 6. Resultados disponíveis: testes do autor e ensaio adicional

### Experimentos realizados pelo autor

O relatório `results/original_tests/report.md` incorpora os resultados de `Data` com as condições do metadata. Ele inclui as comparações em Fashion-MNIST e CIFAR-10, o controle de LR 2×, as varreduras de LR e de transição da sigmoide, as grades de coeficientes de m2 e os contadores do piso.

| Teste do autor | Condições registradas | Resultado recalculado |
|---|---|---|
| AdaSym / AdaSymSigmoid / AdamW | Fashion-MNIST, 100 épocas | Médias na curva: 0,291404 / 0,287134 / 0,302175. Sigmoid tem menor média na curva e nas últimas dez épocas |
| AdaSym / AdamW | LeNet, Fashion-MNIST, 150 épocas, LR 1e-3 | Médias nas últimas dez épocas: 0,038349 / 0,041992 |
| Variante AdaSym nored | Mesmo grupo de arquivos LeNet / Fashion-MNIST, sem redução temporal pelo nome | Média nas últimas dez épocas: 0,038083; ligação exata entre scheduler e arquivos documentada no mapa |
| AdaSymSigmoid / AdamW | CIFAR-10, ResNet-20, 150 épocas, LR 1e-3, WD 1e-2, T=15, piso v/8 | Médias nas últimas dez épocas: 0,015382 / 0,017411 |
| AdamW com LR 2× | LeNet, 150 épocas, dataset descrito de forma diferente entre metadata e slides | Média na curva 0,109575, menor que 0,120462 de AdaSym; no trecho final AdaSym tem média menor |
| LR de AdaSym, 80 pontos | 300 épocas, 10% Fashion-MNIST, AUC, WD 1e-2, valores (0,75; 0,35) | Mínimo 101,313248, índice 54; a grade de LRs não foi fornecida |
| LRs comparativas, 27 pontos | Grade reconstruída do metadata de 1e-4 a 9e-2 | Mínimos 109,020409 e 105,941498; LRs inferidas 0,02 e 0,04 |
| Ponto de transição da sigmoide | LeNet, 20% Fashion-MNIST, 150 épocas, LR 1e-3, WD 1e-2, faixa 0–50 | Menor AUC 121,601784, índice 17 |
| beta3 / beta4 de m2 | Duas grades de 10 × 70 | Mínimos em (0,584341; 0,740568) e (0,674355; 0,406159), como nos slides |
| Ativação do piso v/4 | LeNet, Fashion-MNIST, 150 registros | Razão final c4/full = 9,9777%; c2/full = 14,0758% |
| Divisor do piso, 2–20 | LeNet, 20% Fashion-MNIST, 150 épocas, LR 1e-3, WD 1e-2, sem decaimento | Teste descrito no metadata; arquivo `div_aucloss` não incluído |

Esses experimentos integram a evidência original da pesquisa. As losses menores em várias comparações apoiam investigar a dinâmica do AdaSym. A falta de identificação do split e das sementes limita o alcance das conclusões, sem transformar os testes existentes em experimentos apenas planejados. As notas da auditoria complementam essa apresentação dos resultados.

### Ensaio sintético adicional desta revisão

O experimento incluído treina um MLP tanh 12–24–2 em classificação sintética não linear. A triagem usa duas inicializações. A confirmação usa cinco inicializações novas e novas amostras, mantendo o mesmo gerador da tarefa. As configurações são congeladas antes de avaliar o teste. A seleção do checkpoint usa validação.

Esse ensaio usa a extensão de janela comum e uma reconstrução explicitada do denominador histórico. Ele não executa o código original de m2 e não substitui os experimentos de visão realizados pelo autor.

No ensaio de 35 épocas, a busca escolheu `sqrt(q)`. O AdamW com grade de LR ampliada pelo fator sqrt(c) obteve melhor resultado de teste que as variantes AdaSym. O relatório com médias, dispersão e todos os logs está em `results/synthetic/report.md`. Assim, este ensaio não confirma uma vantagem própria da janela. A melhor LR de várias famílias está na borda da grade, sinal de que a faixa deve ser ampliada na próxima fase, usando outro teste ainda não consultado.

A busca entre cinco expressões consome mais configurações do que o ajuste de uma baseline individual. Por isso, o resultado do vencedor simbólico inclui vantagem de orçamento de seleção. Antes de qualquer alegação comparativa, é necessário comparar orçamento total de ajuste, replicar em outras tarefas e manter um teste final intocado.

## 7. Experimentos que podem decidir a hipótese

O protocolo completo aparece em `docs/protocolo.md`. As comparações centrais são AdamW com LR ajustada, AdamW com LR multiplicada por sqrt(c), AdaSym sem consistência, janela comum linear, janela histórica com momentos distintos e janela com sigmoide. Uma baseline com ganho escalar médio pode aproximar o tamanho médio dos passos de AdaSym e testar o valor da adaptação coordenada.

Medir loss e acurácia de validação, tempo até alvo, taxa de divergência, normas de update, distribuição de q, ganho aplicado e frequência de saturação. Um teto que atua quase sempre transforma a regra em um escalonamento quase constante da LR. Isso enfraquece a explicação por consistência seletiva e precisa ser mostrado.

Reproduzir com logs completos os testes já realizados pelo autor em Fashion-MNIST com LeNet e CIFAR-10 com ResNet-20, e acrescentar ao menos uma tarefa fora de visão. Congelar a regra antes da avaliação de transferência. Os treinamentos originais já existem nos dados; o que permanece futuro é sua reprodução com uma especificação completa e a avaliação das extensões adicionais.

## 8. Critério para uma ideia final

Uma fórmula só deve se tornar candidata final quando superar controles de LR sob orçamento equivalente, repetir o efeito em sementes e tarefas distintas e justificar seu custo adicional. A versão escolhida precisa manter implementação, configurações, logs, falhas e ablações rastreáveis.

A hipótese central do AdaSym permanece a dos slides: **amplificar o passo por uma correção de consistência m2², limitada por um piso relativo v/c e eventualmente reduzida por uma sigmoide**. Os experimentos do autor documentam sua investigação empírica. A janela comum normalizada é uma extensão opcional para estudar essa ideia com uma estatística explicitada. A transformação raiz é uma candidata apenas dessa extensão, surgida no ensaio sintético. A seleção de uma regra final deve combinar a recuperação do código original, os testes já realizados e uma rodada controlada de confirmação.

## 9. Relação com trabalhos anteriores

- **Adam:** fundamenta o uso de primeiro e segundo momentos e bias correction. [Kingma e Ba](https://arxiv.org/abs/1412.6980).
- **AdamW:** define weight decay desacoplado e fornece a baseline de referência. [Loshchilov e Hutter](https://arxiv.org/abs/1711.05101).
- **Symbolic Discovery of Optimization Algorithms:** inspiração para busca de programas e seleção de regras transferíveis, incluindo Lion. A implementação atual é uma enumeração restrita e não replica esse sistema. [Chen et al.](https://arxiv.org/abs/2302.06675).
- **AdaBelief:** relaciona adaptação do passo à discrepância entre gradiente e sua previsão por momentum. É uma comparação relevante para estudar persistência e variabilidade, mas não usa a mesma estatística desta proposta. [Zhuang et al.](https://arxiv.org/abs/2010.07468).
- **Centered RMSprop:** já usa um segundo momento centralizado para normalização. A subtração de média ao quadrado possui precedentes. A novidade de AdaSym precisa ser estabelecida na combinação específica de janela, ganho limitado, busca e evidência empírica. [Documentação oficial do PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html).

Esta contextualização não estabelece prioridade nem prova originalidade. Uma revisão bibliográfica sistemática continua necessária antes de submeter um artigo.
