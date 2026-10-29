"""Finite, typed grammar for a restricted symbolic screening experiment.

No eval, arbitrary code execution, or claim of reproducing the original search.
The dimensional search is reduced to monotone functions h(q) on q in [0,1].
"""
from dataclasses import replace
from .numpy_optimizer import Config


GRAMMAR = {
    "linear": ("q", 1),
    "square": ("q*q", 3),
    "sqrt": ("sqrt(q)", 2),
    "rational": ("2*q/(1+q)", 7),
    "tanh": ("tanh(q)/tanh(1)", 5),
}


def candidates(base=None):
    """Equal-sized LR grid for each expression; cap and window frozen here."""
    base = base or Config()
    for rule, (expression, nodes) in GRAMMAR.items():
        for lr in (3e-4, 1e-3, 3e-3):
            yield {"id": f"{rule}_lr{lr:g}", "expression": expression, "nodes": nodes, "config": replace(base, rule=rule, lr=lr)}
