# AdaSym — experimental research in optimization

English | [Português (Brasil)](README.pt-BR.md)

Research on optimization algorithms for machine learning, developed by **Ítalo Capuzzo** within a machine learning and AI study group. The project combines **symbolic discovery, neural network experiments, and data science** to investigate adaptive update rules.

Loss curves, hyperparameter sweeps, and internal optimizer statistics guide pattern identification and hypothesis refinement. This process accompanies the evolution of AdaSym from a discovered expression to an update rule incorporating directional memory, bounded amplification, and temporal weighting.

[Experiment analysis](results/research/report.md) · [Diagnostic notebook](notebooks/threshold-diagnostics.ipynb) · [Formulation](docs/pesquisa.md) · [Symbolic search on Kaggle](https://www.kaggle.com/code/italocapuzzo/symbolic-optimizer-discovery-on-mnist-via-gp)

## Update rule

AdaSym adjusts the step size to recent gradient consistency at each coordinate. Persistent directional memory reduces the update denominator, producing an **element-wise boost**. The relative floor `v/c` bounds this amplification.

$$d=\max\left(v-s\,m_2^2,\frac{v}{c}\right),\qquad \theta_{k+1}=\theta_k-\eta\frac{m_1}{\sqrt d}.$$

| Component | Role |
|---|---|
| `m1` | Update direction obtained from gradient memory |
| `v` | Normalization scale based on squared gradients |
| `m2` | Directional memory used in the denominator correction |
| `s` | Correction weight, optionally varied during training |
| `c` | Control of the floor and maximum amplification |
| `eta` | Learning rate |

The **AdaSymSigmoid** variant gradually reduces the directional correction through a sigmoid over the training epochs.

## Data analysis behind the formula

The experiments investigate both training outcomes and the optimizer's internal behavior. Loss comparisons examine convergence dynamics; boost counters examine how the rule acts on coordinates; parameter sweeps study the response to hyperparameters.

| Formulation stage | Research focus |
|---|---|
| Expression obtained through symbolic search | Candidate structures for the update rule |
| Squared memory in the denominator | Relationship between directional memory and gradient scale |
| Relative floor `v/c` | Control of step amplification |
| Separate memory `m2` | Response to the correction's own coefficients |
| Sigmoid time weight | Evolution of the correction during training |

This progression reflects an investigation guided by data: observing patterns, comparing variants, and refining the update rule.

## Boost diagnostics and threshold sensitivity

A central analysis uses the `AdaSym_fminist_c4`, `c2`, and `full` records from **150 epochs of LeNet/Fashion-MNIST**, with a `v/4` floor.

The counters track the directional boost produced by the consistency window. `full` is associated with the full boost limited by the threshold; `c4` and `c2` track partial-boost ranges related to `v/4` and `v/2`, with **`c4` included in `c2`**.

Diagnostics examine cumulative counts, per-epoch increments, and composition within `c2`. These records support investigation of temporal changes and the relationship between the monitored boost ranges.

![Boost counter dynamics](results/research/threshold_dynamics.png)

[Visual report](results/research/report.html) · [Per-epoch counters](results/research/threshold_by_epoch.csv) · [Methodology](docs/investigacao_dados.md)

## Experiments and comparisons

| Area | Data and analysis |
|---|---|
| Optimizer comparison | AdaSym, AdaSymSigmoid, and AdamW on Fashion-MNIST and CIFAR-10 |
| Training dynamics | Loss curves, normalized AUC, and late-training behavior |
| Coordinate-wise boost | Cumulative series, increments, and counter composition |
| Hyperparameter sensitivity | Learning rate, memory coefficients, and sigmoid transition |
| Symbolic discovery | Fitness evolution, expression size, and selected formulas |

![Loss curves and experiment comparisons](results/research/loss_diagnostics.png)

The [analysis report](results/research/report.md) brings together tables and interpretations. Results include CSV exports, PNG/SVG figures, and a manifest with input hashes and analysis parameters.

## Symbolic discovery

The **Symbolic Optimizer Discovery on MNIST via GP** notebook searches expressions built from gradients and their moments using genetic programming with DEAP and PyTorch.

- Tournament selection, crossover, mutation, and elitism.
- Candidate evaluation on LeNet/MNIST using validation loss relative to AdamW.
- Expression complexity and depth controls.
- Tracking of fitness and selected formulas.

[Notebook code](notebooks/symbolic-optimizer-discovery-on-mnist-via-gp.ipynb) · [Execution guide](notebooks/README.md)

## Run

Requires Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
# Git Bash on Windows:
source .venv/Scripts/activate
python -m pip install -e ".[plots,docs]"
python scripts/analyze_research.py
```

The command generates the report, figures, and tables in `results/research/`. The `notebooks/threshold-diagnostics.ipynb` notebook provides an executable walkthrough in Jupyter.

```bash
python -m unittest discover -s tests -v
python scripts/run_synthetic.py --epochs 35 --output results/runs/synthetic
```

For PyTorch experiments:

```bash
python -m pip install -e ".[torch]"
python scripts/run_fashion_mnist.py --download --method linear --epochs 30 --seed 101
python scripts/run_fashion_mnist.py --method adamw --epochs 30 --seed 101
```

## Structure

| Folder | Contents |
|---|---|
| `data/original/` | NumPy experiment records |
| `notebooks/` | Symbolic discovery and boost diagnostics |
| `src/optimizer_research/` | NumPy/PyTorch implementations and analysis functions |
| `scripts/` | Analysis, visualization, and experiments |
| `results/research/` | Report, tables, figures, and provenance |
| `docs/` | Formulation and methodology |
| `references/` | Research presentation |
| `tests/` | Verification of implementations and calculations |
