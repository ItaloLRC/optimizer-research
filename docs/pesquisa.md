# AdaSym: descoberta simbólica de otimizadores com janela de consistência

**Versão de pesquisa: 29 de outubro de 2026.** Projeto exploratório. O material original contém slides e arrays de resultados, sem os programas de busca e treinamento. Esta revisão preserva o histórico e propõe uma especificação nova e verificável.

## 1. Pergunta de pesquisa

Uma rede neural pode convergir mais rapidamente quando o otimizador aumenta seletivamente seus passos nas coordenadas cujos gradientes mantêm direção e magnitude durante uma janela recente?

A hipótese do projeto é que a persistência do gradiente informa o quanto se pode amplificar o passo. A regressão simbólica serve para explorar regras candidatas, identificar estruturas promissoras e restringir progressivamente o espaço de busca. A avaliação empírica deve separar um ganho causado pela consistência de um ganho obtido apenas por uma taxa de aprendizado maior.

O objetivo verificável é melhorar a eficiência até uma loss ou acurácia predefinida, mantendo qualidade de teste e orçamento de ajuste comparáveis. Convergência em menos épocas, tempo até o alvo, custo de estado e generalização são resultados diferentes e devem aparecer separadamente.

## 2. O que o material original sustenta

Os slides documentam a sequência:

$$
\frac{m_1}{\sqrt{|v-m_1|}}
\quad\longrightarrow\quad
\frac{m_1}{\sqrt{|v-m_1^2|}}
\quad\longrightarrow\quad
\frac{m_1}{\sqrt{\max(v-m_2^2,v/c)}}
\quad\longrightarrow\quad
\frac{m_1}{\sqrt{\max(v-\alpha m_2^2,v/c)}}.
$$

A primeira expressão subtrai grandezas de ordens diferentes se v representa E[g²] e m1 representa E[g]. A versão com m1² corrige a compatibilidade dimensional. O valor absoluto transforma qualquer diferença negativa em uma normalização positiva, mas sua interpretação depende da diferença entre as janelas. O piso v/c torna explícito um limite relativo do denominador. A sigmoide controla quanto da correção permanece ao longo do treinamento.

As matrizes existentes permitem descrever a loss registrada e localizar valores inválidos. Elas não identificam integralmente sementes, split dos dados, precisão, inicializações, scheduler ou recorrência de m2. Por isso, não demonstram uma nova garantia de convergência, superioridade estatística nem reprodução da descoberta simbólica. O relatório em `results/historical/report.md` quantifica o que pode ser medido e explicita as ambiguidades.

O histórico já contém um controle relevante de escala do passo: nos arquivos de 150 posições, a média registrada é 0,135194 para AdamW, 0,120462 para AdaSym e 0,109575 para o arquivo descrito como AdamW com LR 2×. Nas últimas dez posições, AdaSym tem média menor, 0,038349 contra 0,041124 do controle 2×. Essa diferença entre velocidade inicial e loss tardia ajuda a formular a próxima comparação. O conflito de rótulo MNIST/Fashion-MNIST e a ausência do código impedem tratar esses números como um experimento controlado já estabelecido. O mapa em `docs/mapa_experimentos.md` organiza as condições relatadas, incluindo os testes de LR, sigmoide, piso e betas.

## 3. Janela de consistência

Para cada coordenada i, mantemos uma média exponencial do gradiente e uma média do gradiente ao quadrado, ambas com o mesmo beta:

$$s_t=\beta_w s_{t-1}+(1-\beta_w)g_t,$$
$$u_t=\beta_w u_{t-1}+(1-\beta_w)g_t^2.$$

Com estado inicial zero, usamos a mesma correção de viés nas duas:

$$\widehat s_t=\frac{s_t}{1-\beta_w^t},\qquad \widehat u_t=\frac{u_t}{1-\beta_w^t}.$$

A consistência proposta é

$$q_t=\operatorname{clip}\left(\frac{\widehat s_t^2}{\widehat u_t+\epsilon^2},0,1\right).$$

As operações são coordenada a coordenada. A expressão tem interpretação de momento normalizado: a média mantém seu peso quando os gradientes concordam e sofre cancelamento quando mudam de sinal. Oscilações de magnitude também reduzem q.

Com pesos normalizados não negativos, Jensen garante `(média g)² ≤ média(g²)`. Assim, a razão pertence a [0,1], salvo arredondamento. Usar a mesma janela e a mesma correção de viés é essencial para essa interpretação. A subtração histórica `v − m2²`, com v e m2 calculados em janelas diferentes, não oferece essa propriedade.

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

## 4. Regra de atualização proposta

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

## 6. Evidência nova e interpretação

O experimento incluído treina um MLP tanh 12–24–2 em classificação sintética não linear. A triagem usa duas inicializações. A confirmação usa cinco inicializações novas e novas amostras, mantendo o mesmo gerador da tarefa. As configurações são congeladas antes de avaliar o teste. A seleção do checkpoint usa validação.

No ensaio de 35 épocas, a busca escolheu `sqrt(q)`. O AdamW com grade de LR ampliada pelo fator sqrt(c) obteve melhor resultado de teste que as variantes AdaSym. O relatório com médias, dispersão e todos os logs está em `results/synthetic/report.md`. Assim, este ensaio não confirma uma vantagem própria da janela. A melhor LR de várias famílias está na borda da grade, sinal de que a faixa deve ser ampliada na próxima fase, usando outro teste ainda não consultado.

A busca entre cinco expressões consome mais configurações do que o ajuste de uma baseline individual. Por isso, o resultado do vencedor simbólico inclui vantagem de orçamento de seleção. Antes de qualquer alegação comparativa, é necessário comparar orçamento total de ajuste, replicar em outras tarefas e manter um teste final intocado.

## 7. Experimentos que podem decidir a hipótese

O protocolo completo aparece em `docs/protocolo.md`. As comparações centrais são AdamW com LR ajustada, AdamW com LR multiplicada por sqrt(c), AdaSym sem consistência, janela comum linear, janela histórica com momentos distintos e janela com sigmoide. Uma baseline com ganho escalar médio pode aproximar o tamanho médio dos passos de AdaSym e testar o valor da adaptação coordenada.

Medir loss e acurácia de validação, tempo até alvo, taxa de divergência, normas de update, distribuição de q, ganho aplicado e frequência de saturação. Um teto que atua quase sempre transforma a regra em um escalonamento quase constante da LR. Isso enfraquece a explicação por consistência seletiva e precisa ser mostrado.

Testar Fashion-MNIST com LeNet, CIFAR-10 com ResNet-20 e ao menos uma tarefa fora de visão. Congelar a regra antes da avaliação de transferência. A configuração histórica citada é parte do contexto, não uma reprodução já obtida.

## 8. Critério para uma ideia final

Uma fórmula só deve se tornar candidata final quando superar controles de LR sob orçamento equivalente, repetir o efeito em sementes e tarefas distintas e justificar seu custo adicional. A versão escolhida precisa manter implementação, configurações, logs, falhas e ablações rastreáveis.

O resultado desta revisão é uma hipótese operacional: **amplificar de forma limitada o passo do AdamW segundo a consistência de gradientes em uma janela comum**. A janela linear é o ponto de partida interpretável. A transformação raiz é uma candidata surgida na triagem sintética, ainda sem confirmação de superioridade. As evidências atuais não justificam declarar um otimizador final.

## 9. Relação com trabalhos anteriores

- **Adam:** fundamenta o uso de primeiro e segundo momentos e bias correction. [Kingma e Ba](https://arxiv.org/abs/1412.6980).
- **AdamW:** define weight decay desacoplado e fornece a baseline de referência. [Loshchilov e Hutter](https://arxiv.org/abs/1711.05101).
- **Symbolic Discovery of Optimization Algorithms:** inspiração para busca de programas e seleção de regras transferíveis, incluindo Lion. A implementação atual é uma enumeração restrita e não replica esse sistema. [Chen et al.](https://arxiv.org/abs/2302.06675).
- **AdaBelief:** relaciona adaptação do passo à discrepância entre gradiente e sua previsão por momentum. É uma comparação relevante para estudar persistência e variabilidade, mas não usa a mesma estatística desta proposta. [Zhuang et al.](https://arxiv.org/abs/2010.07468).
- **Centered RMSprop:** já usa um segundo momento centralizado para normalização. A subtração de média ao quadrado possui precedentes. A novidade de AdaSym precisa ser estabelecida na combinação específica de janela, ganho limitado, busca e evidência empírica. [Documentação oficial do PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html).

Esta contextualização não estabelece prioridade nem prova originalidade. Uma revisão bibliográfica sistemática continua necessária antes de submeter um artigo.
