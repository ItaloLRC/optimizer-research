# Symbolic optimizer discovery

English | [Português (Brasil)](README.pt-BR.md)

[Open on Kaggle](https://www.kaggle.com/code/italocapuzzo/symbolic-optimizer-discovery-on-mnist-via-gp) · [Open the notebook on GitHub](symbolic-optimizer-discovery-on-mnist-via-gp.ipynb)

The **Symbolic Optimizer Discovery on MNIST via GP** notebook, by Ítalo Capuzzo, implements an evolutionary search for neural network update formulas. The implementation combines **DEAP, PyTorch, and MNIST** to turn symbolic expressions into executable optimizers and evaluate their training dynamics on a LeNet.

## Structure

1. **Rule representation:** expression trees with inputs `g`, `m_hat`, and `v_hat` — the gradient and bias-corrected moments.
2. **Formula execution:** `SymOptimizer` applies the DEAP-compiled expression to the network parameters.
3. **Fitness:** the `train_one` function computes the area under the validation loss curve; `evalSymbReg` compares it with the AdamW baseline and applies a complexity penalty.
4. **Evolution:** tournament selection, crossover, mutation, elitism, and Hall of Fame.
5. **Visualization:** plots of fitness, tree size, and selected formulas.

The initial population includes expressions inspired by SGD and Adam. Operators include arithmetic operations, protected square root and division, sign functions, and clipping.

## Run on Kaggle

1. Open the [original notebook](https://www.kaggle.com/code/italocapuzzo/symbolic-optimizer-discovery-on-mnist-via-gp) and use **Copy & Edit**.
2. Add the [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data) competition data according to Kaggle's access conditions.
3. Check the `train.csv` path in the `task_mnist` function and select the execution environment.
4. Run the cells in order. `NUM_POPULATION`, `NUM_GENERATIONS`, and `DATA_SIZE` control the search configuration.

The code selects CUDA when a GPU is available and also supports CPU execution.

## Run locally

Install PyTorch for your environment, JupyterLab, and the required libraries:

```bash
python -m pip install torch numpy pandas matplotlib scikit-learn deap jupyterlab
python -m jupyterlab
```

Open `symbolic-optimizer-discovery-on-mnist-via-gp.ipynb`, adjust the `train.csv` path in `task_mnist` to the local data copy, and run the cells in order.

## Source

This copy preserves the code from **version 14** published on Kaggle. The version and file SHA-256 hash are recorded in [source.json](source.json).
