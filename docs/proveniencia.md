# Proveniência dos experimentos

**Pesquisa: Ítalo Capuzzo.** Os experimentos, a análise dos padrões e a evolução da fórmula integram a pesquisa AdaSym.

## Fontes

- `data/original/`: registros NumPy e metadados dos experimentos.
- `data/source_manifest.json`: hashes SHA-256 das fontes.
- `data/experiment_schema.json`: descrição dos contadores do boost.
- `references/AdaSym-original.pptx`: apresentação da pesquisa.
- `docs/slides_extraidos.json`: texto e referências às figuras da apresentação.
- `notebooks/symbolic-optimizer-discovery-on-mnist-via-gp.ipynb`: notebook de busca simbólica; origem em `notebooks/source.json`.

## Análise

O relatório em `results/research/` reúne contadores por época, composição interna de c2, métricas de loss por trajetória e varreduras. O manifesto registra hashes dos dados, versões do ambiente e parâmetros da análise. O notebook apresenta cálculos e gráficos executados.

As implementações de janela comum e os ensaios sintéticos têm suas configurações próprias documentadas no repositório.
