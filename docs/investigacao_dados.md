# Ciência de dados como metodologia da pesquisa AdaSym

**Autor da pesquisa: Ítalo Capuzzo.** Os experimentos, a análise dos padrões e a evolução da fórmula de atualização fazem parte do trabalho do autor. A descrição específica do projeto é **pesquisa experimental em algoritmos de otimização para machine learning**; ciência de dados caracteriza a metodologia usada para investigar os resultados e revisar hipóteses.

## Da observação à revisão da regra

O trabalho do autor reúne busca de expressões, comparação de losses, exploração de hiperparâmetros e diagnóstico interno do otimizador. Os gráficos servem para responder perguntas: quando a regra amplifica o passo, quanto o limitador atua, como o comportamento muda no treino e quais diferenças permanecem quando a baseline usa uma LR maior.

A sequência dos slides 4–7 documenta a passagem da expressão encontrada para o quadrado da memória, a introdução do piso relativo, a memória separada e a sigmoide. Ítalo relata que suas conclusões após analisar os dados e identificar padrões participaram desse desenvolvimento. Os arquivos não estabelecem uma cronologia causal completa entre cada gráfico e cada mudança; essa relação não deve ser inventada.

| Parte da investigação do autor | Uso dos dados | Relação com a formulação |
|---|---|---|
| Curvas de loss e controle LR 2× | Examinar rapidez inicial e comportamento tardio | Questionar quanto do resultado vem da escala do passo |
| Contadores c4/c2/full | Medir incidência e mudança temporal do piso | Investigar saturação e seletividade do limitador |
| Grades beta3/beta4 | Examinar objetivo e regiões inválidas | Investigar a parametrização da memória m2 |
| Varredura da sigmoide | Comparar valores do objetivo em pontos de transição | Investigar por quanto tempo manter a correção |
| Varreduras de LR | Comparar respostas ao ajuste do tamanho do passo | Separar configuração promissora de conclusão geral |

Isso é diagnóstico interno do **otimizador**. Os contadores disponíveis não são ativações de neurônios, mapas de atenção ou medidas por camada; não permitem atribuir o efeito a uma região específica da rede.

## Experimento central: piso v/4

O autor confirmou em 13 de novembro de 2026:

- São **150 épocas**; “1550” no metadata é um erro de digitação.
- As contagens são de avaliações de gradientes/parâmetros afetadas pelo threshold, acumuladas durante o treinamento.
- `c4` corresponde a ≤ v/4, `c2` a ≤ v/2 e `full` ao total.

A confirmação está registrada em [author_clarifications.json](../data/author_clarifications.json). Os arquivos originais permanecem intactos.

As três séries são acumuladas e aninhadas. A análise usa diferenças sucessivas para obter contagens por intervalo. A partição sem dupla contagem é `Δc4`, `Δc2 − Δc4` e `Δfull − Δc2`. As taxas de uma janela são razões das somas, ponderadas pelas avaliações, e não médias simples de percentuais.

O primeiro intervalo tem aproximadamente metade da exposição dos seguintes. A origem dessa diferença não é conhecida. Não se atribui a ela uma mudança de batch size, arquitetura ou instrumentação sem recuperar o código.

## Perguntas e limites das conclusões

A frequência de v/4 permite examinar a hipótese de saturação quase universal e os incrementos permitem examinar a hipótese de incidência constante. O intervalo entre v/4 e v/2 mede a massa adicional atingida pelo segundo limite na mesma trajetória. Nenhuma dessas medidas, isoladamente, escolhe o melhor threshold ou determina seu efeito na acurácia.

Para investigar outras posições do threshold, o relatório usa limites compatíveis com os dois pontos observados. Não interpola uma distribuição inexistente e não confunde esse exercício com um treinamento contrafactual. As barras de limites não são intervalos de confiança.

São avaliações repetidas, correlacionadas no tempo e entre parâmetros. Não se calcula significância estatística usando o total de avaliações como tamanho de uma amostra independente. Também não se presume correspondência entre esses contadores e uma linha específica dos arquivos de loss.

## Reprodução

Execute `python scripts/analyze_research.py` após instalar `.[plots,docs]`. O [relatório](../results/research/report.md) reúne resultados e interpretações; o [notebook](../notebooks/threshold-diagnostics.ipynb) permite percorrer a investigação. Os CSVs mantêm contagens, taxas, janelas, métricas por trajetória e limites de sensibilidade. O manifesto registra os hashes dos dados e do código de análise.

As extensões de janela comum e o ensaio sintético já existentes no repositório têm origem e condições próprias, explicitadas em `pesquisa.md`. Não são usados como evidência histórica da evolução da fórmula original do autor.
