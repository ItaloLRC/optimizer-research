# Proveniência e escopo

## Material recebido

- `references/AdaSym-original.pptx`: cópia integral da apresentação original com 20 slides.
- `data/original/`: cópia dos 16 arrays NumPy, `metadata.txt` e `teste.py` originais.
- `data/source_manifest.json`: hashes SHA-256 das fontes copiadas, sem caminhos pessoais.
- `docs/slides_extraidos.json`: texto e mapeamento das imagens dos slides para consulta. As fórmulas originais estão em imagens, não em texto editável.
- `notebooks/symbolic-optimizer-discovery-on-mnist-via-gp.ipynb`: cópia da versão 14 do notebook público do autor no Kaggle, adicionada a seu pedido. O arquivo `notebooks/source.json` registra sua origem e hash SHA-256.

Os documentos são evidência da pesquisa. Seu conteúdo não foi tratado como instrução para executar programas ou publicar dados. `teste.py` não foi executado nem modificado. As fontes originais em `optim` permanecem intactas.

## Reconstruções novas

A recorrência de m2 não acompanha o material. Os nomes beta3 e beta4 aparecem numa varredura, sem definição da fórmula. A implementação `historical` é uma reconstrução que adota uma EMA normalizada para s e bias correction em todos os momentos. Ela explicita uma possível interpretação e não reivindica equivalência ao programa original.

A janela comum, a gramática pequena, o warmup, a implementação NumPy/PyTorch, os testes e o MLP sintético são adições desta revisão. Os testes novos têm logs próprios. As evidências históricas e novas não foram mescladas em uma estatística única.

`results/original_tests/report.md` apresenta os experimentos realizados pelo autor, com as condições transcritas do metadata e números recalculados dos arrays. O script `summarize_author_experiments.py` gera essa apresentação e seus gráficos sem treinar novamente as redes. A fórmula `d = max(v − s·m2², v/c)` é a do AdaSym dos slides. A razão normalizada q e a função h(q) são identificadas como extensões opcionais, sem atribuí-las aos experimentos originais.

## Dados que ainda precisam ser recuperados

1. Identificação das execuções de descoberta simbólica associadas a cada experimento histórico. O notebook público de programação genética e sua função de fitness estão disponíveis em `notebooks/`.
2. Implementação do otimizador histórico, recorrência de m2, beta4 e bias correction.
3. Código dos treinamentos, curvas de validação e acurácia, sementes e checkpoints.
4. Confirmação da ordem inferida das linhas de `losses_FashionMNIST.npy` e identificação das demais trajetórias.
5. Grade de 80 LRs de `lrs_AdaSym_aucloss.npy` e scheduler de cada comparação.
6. Definição e periodicidade dos contadores c2/c4/full.

Sem esses itens, repetir exatamente os números dos slides seria uma alegação incorreta.

## Autoria, licença e publicação

A pesquisa e os arquivos recebidos pertencem a seus respectivos autores. Nenhuma licença aberta foi escolhida em nome do autor. O notebook público do Kaggle e as atualizações de apresentação foram adicionados ao GitHub a pedido do autor.
