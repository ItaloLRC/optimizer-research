"""Screen symbolic rules on validation; evaluate frozen rules on fresh seeds.

python scripts/run_synthetic.py --output results/synthetic
The test split is only evaluated during confirmation, after ranking is frozen.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from optimizer_research import Config
from optimizer_research.symbolic import candidates
from optimizer_research.synthetic import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/synthetic'))
    parser.add_argument('--epochs', type=int, default=35)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    base = Config()
    ranking = []
    # Same 2 seeds and same 3 learning rates for each symbolic family.
    for candidate in candidates(base):
        runs = [train(candidate['config'], seed, 2026, epochs=args.epochs) for seed in (11, 22)]
        ranking.append(dict(id=candidate['id'], expression=candidate['expression'], nodes=candidate['nodes'], config=candidate['config'].dictionary(), score=float(np.mean([r['val_auc'] for r in runs])), runs=runs))
    ranking.sort(key=lambda r: (r['score'], r['nodes']))
    # Baselines each get exactly 3 LR trials x 2 screening seeds.
    baselines = {}
    for name, config in {'adamw_tuned': replace(base, alpha=0), 'adamw_scaled_lr': replace(base, alpha=0), 'historical': replace(base, rule='historical'), 'adasym_linear': base}.items():
        trials = []
        for lr in (3e-4, 1e-3, 3e-3):
            config_lr = replace(config, lr=lr*(np.sqrt(base.cap) if name == 'adamw_scaled_lr' else 1))
            scores = [train(config_lr, seed, 2026, epochs=args.epochs)['val_auc'] for seed in (11, 22)]
            trials.append((float(np.mean(scores)), config_lr))
        baselines[name] = min(trials, key=lambda x: x[0])[1]
    selected = dict(baselines, symbolic_selected=Config(**ranking[0]['config']))
    # This deliberately includes fresh data and 5 fresh initialization seeds.
    confirmation = {}
    for name, config in selected.items():
        confirmation[name] = [train(config, seed, 90210, epochs=args.epochs, include_test=True) for seed in (101, 202, 303, 404, 505)]
    payload = dict(environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()), epochs=args.epochs, screening_seeds=[11,22], confirmation_seeds=[101,202,303,404,505], screening_task_seed=2026, confirmation_task_seed=90210, selection_metric='mean validation cross entropy across epochs', ranking=ranking, frozen_configs={k:v.dictionary() for k,v in selected.items()}, confirmation=confirmation, elapsed_seconds=time.perf_counter()-start)
    (args.output/'runs.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Experimento novo em MLP sintético', '', 'Classificação não linear, 900 amostras de treino, 250 de validação e 250 de teste. MLP tanh 12–24–2, minibatch 64. Valores abaixo são média ± desvio padrão amostral entre 5 inicializações. O conjunto de teste é o mesmo nas 5 inicializações.', '', f"Épocas: {args.epochs}. Fórmula selecionada na triagem: `{ranking[0]['expression']}`. A triagem usou outras sementes e outras amostras.", '', '| Método | LR congelada | Test loss | Test accuracy | AUC val média por época |', '|---|---:|---:|---:|---:|']
    for name, runs in confirmation.items():
        stats = []
        for key in ('test_loss', 'test_accuracy', 'val_auc'):
            values = [r[key] for r in runs]
            stats.append(f'{np.mean(values):.5f} ± {np.std(values, ddof=1):.5f}')
        lines.append(f"| {name} | {selected[name].lr:g} | {' | '.join(stats)} |")
    lines += ['', '## Limites da interpretação', '', 'Este teste verifica a implementação e fornece evidência limitada sobre esta tarefa sintética. A seleção simbólica consumiu 15 configurações, enquanto cada baseline usou 3. A comparação da melhor família simbólica contra uma baseline inclui esse orçamento adicional e não prova superioridade em orçamento igual. A confirmação usa apenas uma tarefa nova, 5 inicializações e amostras do mesmo gerador. Não substitui Fashion-MNIST, CIFAR-10 ou avaliação de transferência entre domínios.', '', 'Os modelos de teste usam o checkpoint de menor loss de validação. As configurações não são escolhidas pelo teste. O desvio padrão descreve variação entre inicializações, não incerteza entre datasets nem um intervalo de confiança.', '', 'Os métodos históricos são reconstruções novas com momentos explicitados. Não reproduzem o treinamento original ausente. A formulação linear com janela comum constitui a hipótese principal, mesmo se outra transformação vencer a triagem. Não se escolhe uma ideia final com base neste único experimento.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines[:15]))
    print(f'Elapsed: {time.perf_counter()-start:.1f}s')


if __name__ == '__main__':
    main()
