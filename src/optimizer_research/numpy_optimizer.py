"""NumPy reference for AdaSym consistency windows.

All moments use bias correction. Parameters are float64 arrays updated in place.
The historical branch is a documented reconstruction, not recovered original code.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np


def effective_window(beta: float, steps: int | None = None) -> float:
    """Effective sample size of normalized exponential weights, not a hard window."""
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    if steps is not None and steps < 1:
        raise ValueError("steps must be positive")
    asymptotic = (1 + beta) / (1 - beta)
    return asymptotic if steps is None else asymptotic * (1 - beta**steps) / (1 + beta**steps)


def coherence(mean, second, eps=1e-8):
    """Matched-window E[g]^2/E[g^2], coordinatewise in [0,1]."""
    return np.clip(np.square(mean) / (second + eps**2), 0.0, 1.0)


@dataclass(frozen=True)
class Config:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    beta_window: float = 0.9
    cap: float = 4.0
    alpha: float = 1.0
    eps: float = 1e-8
    weight_decay: float = 1e-2
    rule: str = "linear"
    warmup_steps: int = 20
    schedule: str = "constant"
    midpoint: float = 0.3
    temperature: float = 0.05

    def __post_init__(self):
        for name in ("lr", "beta1", "beta2", "beta_window", "cap", "alpha", "eps", "weight_decay", "midpoint", "temperature"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.lr <= 0 or self.eps <= 0 or self.cap < 1 or self.weight_decay < 0:
            raise ValueError("lr/eps > 0, cap >= 1, weight_decay >= 0 required")
        if self.lr * self.weight_decay > 1:
            raise ValueError("lr * weight_decay must be <= 1")
        if not all(0 <= b < 1 for b in (self.beta1, self.beta2, self.beta_window)):
            raise ValueError("EMA betas must be in [0, 1)")
        if not 0 <= self.alpha <= 1 or not 0 <= self.midpoint <= 1 or self.temperature <= 0:
            raise ValueError("invalid alpha or schedule")
        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        if self.rule not in {"linear", "square", "sqrt", "rational", "tanh", "historical"}:
            raise ValueError("unknown rule")
        if self.schedule not in {"constant", "sigmoid"}:
            raise ValueError("unknown schedule")

    def dictionary(self):
        return asdict(self)


def transform(q, rule):
    if rule in {"linear", "historical"}:
        return q
    if rule == "square":
        return q * q
    if rule == "sqrt":
        return np.sqrt(q)
    if rule == "rational":
        return 2 * q / (1 + q)
    if rule == "tanh":
        return np.tanh(q) / np.tanh(1.0)
    raise ValueError("unknown transform")


class AdaSym:
    """Optimizer with a bounded amplification of the AdamW adaptive update.

    states: Adam first/second moments and window first/second moments.
    A None gradient skips both parameter decay and its step counter.
    """
    def __init__(self, parameters, config=None):
        self.config = config or Config()
        self.parameters = list(parameters)
        if not self.parameters or any(p.dtype != np.float64 for p in self.parameters):
            raise ValueError("provide nonempty float64 parameter arrays")
        self.states = [dict(t=0, m=np.zeros_like(p), v=np.zeros_like(p), s=np.zeros_like(p), u=np.zeros_like(p)) for p in self.parameters]
        self.diagnostics = {}

    def step(self, gradients, progress=None):
        c = self.config
        gradients = list(gradients)
        if len(gradients) != len(self.parameters):
            raise ValueError("one gradient per parameter required")
        if c.schedule == "sigmoid" and (progress is None or not np.isfinite(progress) or not 0 <= progress <= 1):
            raise ValueError("sigmoid schedule needs progress in [0,1]")
        # Validate the entire step before mutating any state.
        for p, g in zip(self.parameters, gradients):
            if g is not None and (g.shape != p.shape or not np.isfinite(g).all()):
                raise ValueError("gradient shape mismatch or nonfinite gradient")
        qs, gains, floors, sizes = [], [], [], []
        for p, g, st in zip(self.parameters, gradients, self.states):
            if g is None:
                continue
            st['t'] += 1
            t = st['t']
            for name, beta, value in (("m", c.beta1, g), ("v", c.beta2, g*g), ("s", c.beta_window, g), ("u", c.beta_window, g*g)):
                st[name] *= beta
                st[name] += (1 - beta) * value
            m = st['m'] / (1 - c.beta1**t)
            v = st['v'] / (1 - c.beta2**t)
            s = st['s'] / (1 - c.beta_window**t)
            u = st['u'] / (1 - c.beta_window**t)
            q = coherence(s, u, c.eps)
            alpha = c.alpha * (min(1.0, t / c.warmup_steps) if c.warmup_steps else 1.0)
            if c.schedule == "sigmoid":
                z = np.clip((c.midpoint - progress) / c.temperature, -60, 60)
                alpha *= 1 / (1 + np.exp(-z))
            if c.rule == "historical":
                raw = v - alpha * s*s
                d = np.maximum(raw, v / c.cap)
            else:
                raw = 1 - alpha * transform(q, c.rule)
                d = v * np.maximum(raw, 1 / c.cap)
            denom = np.sqrt(d) + c.eps
            gain = (np.sqrt(v) + c.eps) / denom
            p *= 1 - c.lr * c.weight_decay
            p -= c.lr * m / denom
            qs.append(float(q.mean()))
            gains.append(float(gain.mean()))
            floors.append(float(np.mean(d <= v / c.cap)))
            sizes.append(p.size)
        if sizes:
            self.diagnostics = {"coherence": float(np.average(qs, weights=sizes)), "gain": float(np.average(gains, weights=sizes)), "floor_fraction": float(np.average(floors, weights=sizes))}

    def state_dict(self):
        return {"config": self.config.dictionary(), "states": [{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in st.items()} for st in self.states]}
