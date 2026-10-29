"""Plot actual new runs and an analytic consistency example."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from optimizer_research import AdaSym, Config

root = Path(__file__).resolve().parents[1]
out = root/'results/synthetic'
payload = json.loads((out/'runs.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, ax = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for name, runs in payload['confirmation'].items():
    a = np.array([[r['val_loss'] for r in run['curve']] for run in runs])
    x = np.arange(1,a.shape[1]+1)
    ax[0].plot(x,a.mean(axis=0),label=name)
    ax[0].fill_between(x,a.mean(axis=0)-a.std(axis=0,ddof=1),a.mean(axis=0)+a.std(axis=0,ddof=1),alpha=.10)
ax[0].set(xlabel='Época',ylabel='Cross entropy de validação',title='MLP sintético: média ± DP entre inicializações')
ax[0].legend(fontsize=7)
for name, seq in [('Constante',np.ones(120)),('Alternante',(-1.)**np.arange(120)),('Inversão no passo 60',np.r_[np.ones(60),-np.ones(60)])]:
    o = AdaSym([np.zeros(1)],Config(beta_window=.9,warmup_steps=20))
    q = []
    for g in seq:
        o.step([np.array([g])]); q.append(o.diagnostics['coherence'])
    ax[1].plot(np.arange(1,121),q,label=name)
ax[1].set(xlabel='Passo',ylabel='Consistência q',title='Resposta da janela a sinais controlados',ylim=(-.03,1.03))
ax[1].legend(fontsize=8)
fig.savefig(out/'summary.png',dpi=180)
