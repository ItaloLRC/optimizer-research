# Descoberta simbólica de otimizadores

[Abrir no Kaggle](https://www.kaggle.com/code/italocapuzzo/symbolic-optimizer-discovery-on-mnist-via-gp) · [Abrir o notebook no GitHub](symbolic-optimizer-discovery-on-mnist-via-gp.ipynb)

O notebook **Symbolic Optimizer Discovery on MNIST via GP**, de Ítalo Capuzzo, implementa uma busca evolutiva por fórmulas de atualização de redes neurais. A implementação combina **DEAP, PyTorch e MNIST** para transformar expressões simbólicas em otimizadores executáveis e avaliar sua dinâmica de treinamento em uma LeNet.

## Organização

1. **Representação das regras:** árvores de expressões com entradas `g`, `m_hat` e `v_hat` — gradiente e momentos com correção de viés.
2. **Execução das fórmulas:** `SymOptimizer` aplica a expressão compilada pelo DEAP aos parâmetros da rede.
3. **Fitness:** a função `train_one` calcula a área sob a curva de perda de validação; `evalSymbReg` compara esse valor ao baseline AdamW e aplica uma penalização por complexidade.
4. **Evolução:** seleção por torneio, cruzamento, mutação, elitismo e Hall of Fame.
5. **Visualização:** gráficos do fitness, tamanho das árvores e fórmulas selecionadas.

A população inicial inclui expressões inspiradas em SGD e Adam. Os operadores incluem operações aritméticas, raiz e divisão protegidas, funções de sinal e clipping.

## Executar no Kaggle

1. Abra o [notebook original](https://www.kaggle.com/code/italocapuzzo/symbolic-optimizer-discovery-on-mnist-via-gp) e use **Copy & Edit**.
2. Adicione os dados da competição [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data), conforme as condições de acesso do Kaggle.
3. Confira o caminho de `train.csv` na função `task_mnist` e selecione o ambiente de execução.
4. Execute as células em ordem. `NUM_POPULATION`, `NUM_GENERATIONS` e `DATA_SIZE` controlam a configuração da busca.

O código seleciona CUDA quando uma GPU está disponível e também possui caminho de execução em CPU.

## Executar localmente

Instale PyTorch adequado ao seu ambiente, JupyterLab e as bibliotecas utilizadas:

```bash
python -m pip install torch numpy pandas matplotlib scikit-learn deap jupyterlab
python -m jupyterlab
```

Abra `symbolic-optimizer-discovery-on-mnist-via-gp.ipynb`, ajuste o caminho de `train.csv` em `task_mnist` para a cópia local dos dados e execute as células em ordem.

## Origem

Esta cópia preserva o código da **versão 14** publicada no Kaggle. A versão e o hash SHA-256 do arquivo estão registrados em [source.json](source.json).
