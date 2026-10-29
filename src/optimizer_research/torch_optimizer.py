"""PyTorch implementation of the proposed consistency window, with FP32 state.

Dense real gradients only. No fused, differentiable, or capturable mode.
Set alpha=0 or cap=1 to obtain AdamW with matching moment hyperparameters.
"""
import math
import torch
from torch.optim import Optimizer
from .numpy_optimizer import Config


class AdaSymTorch(Optimizer):
    def __init__(self, params, config=None):
        c = config or Config()
        super().__init__(params, c.dictionary())
        self.progress = None
        self.diagnostics = {}

    def set_progress(self, value):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("progress must be in [0, 1]")
        self.progress = value

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Reject invalid gradients before updating any parameter.
        for group in self.param_groups:
            if group['schedule'] == 'sigmoid' and self.progress is None:
                raise ValueError("call set_progress before a sigmoid step")
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse or p.is_complex() or not p.is_floating_point():
                        raise ValueError("only dense real floating parameters supported")
                    if not torch.isfinite(p.grad).all():
                        raise ValueError("nonfinite gradient")
        metrics = []
        for group in self.param_groups:
            c = group
            for p in group['params']:
                if p.grad is None:
                    continue
                dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
                g = p.grad.to(dtype)
                st = self.state[p]
                if not st:
                    st['t'] = 0
                    for name in ('m', 'v', 's', 'u'):
                        st[name] = torch.zeros_like(p, dtype=dtype)
                st['t'] += 1
                t = st['t']
                for name, beta, value in (('m', c['beta1'], g), ('v', c['beta2'], g*g), ('s', c['beta_window'], g), ('u', c['beta_window'], g*g)):
                    st[name].mul_(beta).add_(value, alpha=1-beta)
                m = st['m'] / (1-c['beta1']**t)
                v = st['v'] / (1-c['beta2']**t)
                s = st['s'] / (1-c['beta_window']**t)
                u = st['u'] / (1-c['beta_window']**t)
                q = (s.square()/(u+c['eps']**2)).clamp(0, 1)
                alpha = c['alpha'] * (min(1., t/c['warmup_steps']) if c['warmup_steps'] else 1.)
                if c['schedule'] == 'sigmoid':
                    z = max(-60, min(60, (c['midpoint']-self.progress)/c['temperature']))
                    alpha *= 1/(1+math.exp(-z))
                rule = c['rule']
                if rule == 'historical':
                    d = torch.maximum(v-alpha*s.square(), v/c['cap'])
                else:
                    h = {'linear': lambda: q, 'square': lambda: q.square(), 'sqrt': lambda: q.sqrt(), 'rational': lambda: 2*q/(1+q), 'tanh': lambda: q.tanh()/math.tanh(1.)}[rule]()
                    d = v*torch.clamp(1-alpha*h, min=1/c['cap'])
                denom = d.sqrt()+c['eps']
                p.mul_(1-c['lr']*c['weight_decay'])
                p.add_((-c['lr']*m/denom).to(p.dtype))
                metrics.append((p.numel(), q.mean().item(), ((v.sqrt()+c['eps'])/denom).mean().item(), (d <= v/c['cap']).float().mean().item()))
        if metrics:
            count = sum(row[0] for row in metrics)
            self.diagnostics = {name: sum(row[0]*row[i] for row in metrics)/count for i, name in enumerate(('coherence', 'gain', 'floor_fraction'), 1)}
        return loss
