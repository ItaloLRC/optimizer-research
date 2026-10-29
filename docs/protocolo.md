# Protocolo de continuação

## Hipóteses registradas antes da próxima rodada

H1: a consistência coordenada reduz o tempo até uma qualidade alvo em comparação com AdamW ajustado. H2: esse efeito permanece quando AdamW pode usar taxas maiores e o orçamento total de ajuste é equivalente. H3: uma janela comum é mais interpretável e ao menos tão robusta quanto a subtração histórica com janelas distintas. H4: a janela transfere para tarefas que não participaram da seleção simbólica.

O teste sintético já observado é exploratório. Qualquer modificação baseada nele inicia uma nova rodada e precisa de novos dados finais de confirmação.

## Fases e separação dos dados

| Fase | Uso permitido | Saída congelada |
|---|---|---|
| Busca | Treino e validação em tarefas proxy, 3 sementes por configuração | Fronteira de fórmulas e logs da busca |
| Seleção | Validação com 5 sementes novas e tarefas proxy adicionais | Uma fórmula e regra de ajuste de hiperparâmetros |
| Confirmação | Treino/validação com 10 sementes e configuração congelada | Comparação principal e suas falhas |
| Transferência | Novos datasets e arquiteturas que não guiaram a fórmula | Evidência de generalidade ou seus limites |

Separar as sementes de ordem de minibatch, inicialização, divisão dos dados e busca simbólica. Teste oficial só pode servir à avaliação final. A escolha de época/checkpoint usa validação. Evitar ajustar repetidamente a fórmula após ler resultados do mesmo teste.

## Orçamento justo

Definir antes da rodada quantos treinamentos ou GPU-horas cada família pode usar. Incluir no custo de AdaSym a seleção entre expressões e não apenas o melhor treinamento. Com orçamento pequeno, comparar uma família congelada de cada vez e reservar a busca ampla para uma análise separada.

O ensaio sintético incluído usa três LRs por família e duas sementes. Há cinco famílias simbólicas: seu custo total de seleção é maior que o de uma única baseline. Todas as escolhas e os resultados ficam em `results/synthetic/runs.json`.

Próxima grade sugerida: LRs logarítmicas entre 1e-4 e 3e-2, expandindo se o melhor ponto estiver na borda. Fixar como essa expansão será decidida pela validação antes de tocar o teste. Ajustar weight decay e scheduler de maneira equivalente. Mesmo custo por tentativa e mesmas regras de early stopping.

## Controles e ablações

| Comparação | Pergunta |
|---|---|
| AdamW, alpha=0 | A consistência melhora a baseline dentro da mesma implementação? |
| AdamW com LR maior e bem ajustada | O ganho é apenas escala do passo? |
| AdamW com scheduler equivalente ao ganho médio | O efeito depende do tempo, sem seletividade por coordenada? |
| AdaBelief e centered RMSprop | A estatística proposta agrega algo sobre adaptação por variabilidade? |
| SGD momentum e Lion | A conclusão depende de comparar apenas com AdamW? |
| h(q)=q contra sqrt(q), q² e demais | Qual forma de resposta tem suporte? |
| Janela comum contra v−s² histórico | O benefício exige momentos de janelas diferentes? |
| beta_window: 0,75 / 0,9 / 0,95 / 0,99 | Qual escala temporal captura consistência útil? |
| c: 1 / 2 / 4 / 8 | O ganho sobrevive ao controle da amplificação máxima? |
| Warmup: 0 / 20 / 100 passos | Quanto do efeito vem do viés de consistência inicial? |
| Alpha constante contra sigmoide | A persistência observada basta ou o efeito depende do decaimento temporal? |
| Coordenada contra camada/bloco | A seletividade local importa? |

Esses controles são um plano. O experimento entregue executa apenas as famílias enumeradas em seu relatório. Não são testes já concluídos em visão.

## Métricas e comparação

Fixar uma métrica primária, como tempo até uma acurácia de validação alvo, com a qualidade final de teste como condição de não inferioridade. Se não houver alvo justificável, usar AUC de loss de validação normalizada pelo horizonte e apresentar também curvas completas. Não confundir soma das losses com integral trapezoidal. Declarar o eixo em passos, exemplos ou segundos.

Registrar média, desvio padrão e diferenças pareadas entre sementes. No relatório final, estimar IC por bootstrap pareado sobre repetições independentes, com a unidade de reamostragem explícita. Dez sementes ainda não estimam variabilidade entre datasets. Não aplicar teste t independente a observações autocorrelacionadas de épocas nem tratar milhares de minibatches como milhares de replicações.

Declarar todas as falhas. Reportar não finitos, runs interrompidas, saturação do piso, passos máximos, normas de parâmetros e custo de memória. Não excluir runs divergentes só da média do método. Em buscas grandes, avaliar o vencedor em sementes e tarefas independentes para reduzir viés de seleção.

## Registro por execução

Salvar commit do código, configuração completa, sementes separadas, dataset/split, hash da divisão, arquitetura, inicialização, batch size, precisão, dispositivo, versões, scheduler, épocas, passos, loss/accuracy por split, duração, estado do otimizador, checkpoint escolhido e motivo de falha. Incluir a expressão simbólica canônica, complexidade e histórico de seleção.

## Gate para promoção de uma fórmula

Promover somente se ela melhorar a métrica primária em mais de uma tarefa, mantiver a qualidade final predefinida, superar controles de LR com orçamento equivalente e apresentar custo compatível com o ganho. Se a evidência falhar, publicar o resultado negativo e restringir a hipótese ao domínio onde houver suporte.
