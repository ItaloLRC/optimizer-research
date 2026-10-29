"""Descriptive convergence and late-loss variation in the recorded experiments.

Targets and the final window are chosen after observing these curves. They are
exploratory summaries, not preregistered endpoints or tests of equivalence.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'fashion_adamw': 'AdamW_losses.npy',
    'fashion_adasym': 'AdaSym_losses.npy',
    'fashion_nored': 'AdaSym_loss_fmnist_nored.npy',
    'fashion_adamw_2x': 'Adam_losses2x.npy',
    'cifar_adamw': 'AdamW_losses_cifar10.npy',
    'cifar_adasym_sigmoid': 'AdaSymSig_losses_cifar10.npy',
}


def first_sustained(curve, target=.1, width=5):
    for index in range(len(curve)-width+1):
        if np.all(curve[index:index+width] <= target):
            return index+1
    return None


def main():
    results={}
    for key, filename in FILES.items():
        a=np.load(ROOT/'data/original'/filename,allow_pickle=False)
        if a.shape != (3,150) or not np.isfinite(a).all():
            raise ValueError(f'Unexpected or nonfinite curve: {filename}')
        final=a[:,-20:]
        results[key]=dict(file=filename,target_epoch_mean_curve=first_sustained(a.mean(0)),target_epoch_per_row=[first_sustained(row) for row in a],last20_mean=float(final.mean()),last20_temporal_sd_mean=float(final.std(1,ddof=1).mean()),last20_successive_change_rms=float(np.sqrt(np.mean(np.diff(final,axis=1)**2))))
    out=ROOT/'results/original_tests'
    out.mkdir(exist_ok=True)
    payload=dict(target=.1,consecutive_epochs=5,final_window=20,design='post hoc exploratory description; no equivalence test; row seeds and data split unrecorded',results=results)
    (out/'convergence.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Convergência e oscilação da loss no final do treino','', 'A comparação considera as curvas salvas de 150 épocas. O tempo até o alvo é a primeira época de uma sequência de cinco épocas em que a média das três trajetórias permanece em loss ≤ 0,1. A medida usa épocas de treinamento, sem estimar tempo de execução.', '', 'A variabilidade final é a média dos desvios padrão temporais, calculados separadamente em cada trajetória nas últimas 20 épocas, com ddof=1. Ela inclui qualquer tendência residual desse trecho. A medida descreve a oscilação da loss e não equivale à dispersão entre sementes ou a uma garantia de estabilidade do otimizador.', '', '| Experimento | Método | Época até o alvo na curva média | Loss média nas últimas 20 épocas | DP temporal médio nas últimas 20 épocas |','|---|---|---:|---:|---:|']
    for key,task,label in [('fashion_adamw','Fashion-MNIST / LeNet','AdamW'),('fashion_adasym','Fashion-MNIST / LeNet','AdaSym'),('cifar_adamw','CIFAR-10 / ResNet-20','AdamW'),('cifar_adasym_sigmoid','CIFAR-10 / ResNet-20','AdaSymSigmoid'),('fashion_nored','Grupo LeNet, variante sem redução temporal','AdaSym nored'),('fashion_adamw_2x','Grupo LeNet, rótulo de dataset divergente','AdamW LR 2×')]:
        r=results[key]
        lines.append(f"| {task} | {label} | {r['target_epoch_mean_curve']} | {r['last20_mean']:.6f} | {r['last20_temporal_sd_mean']:.6f} |")
    f=results['fashion_adasym'];fa=results['fashion_adamw'];c=results['cifar_adasym_sigmoid'];ca=results['cifar_adamw']
    lines += ['', '## Interpretação', '', f"No par Fashion-MNIST/LeNet descrito com LR 1e-3, AdaSym atinge o alvo na época {f['target_epoch_mean_curve']} e AdamW na {fa['target_epoch_mean_curve']}. No par CIFAR-10/ResNet-20, ambos com LR 1e-3, AdaSymSigmoid atinge o alvo na época {c['target_epoch_mean_curve']} e AdamW na {ca['target_epoch_mean_curve']}. As variantes AdaSym dessas duas comparações apresentam loss média menor e DP temporal ligeiramente menor nas últimas 20 épocas.", '', 'Essas observações apoiam a hipótese de que AdaSym acelera a convergência em número de épocas com oscilação final da loss comparável à de AdamW, nas configurações avaliadas. Elas não demonstram equivalência estatística de estabilidade nem superioridade em tarefas não avaliadas.', '', 'A conclusão tem exceções importantes: o controle AdamW com LR 2× alcança o alvo antes de AdaSym, e a variante AdaSym nored apresenta DP temporal maior no final. A alegação de aceleração se refere aos pares descritos com a mesma LR; o comportamento depende da variante e do ajuste dos hiperparâmetros.', '', 'O alvo de 0,1, as cinco épocas consecutivas e a janela final de 20 épocas foram escolhidos após observar as curvas, para produzir uma descrição auditável. Não constituem um protocolo pré-registrado. Os IDs de sementes e o split da loss não acompanham os arquivos; há ainda uma ambiguidade de scheduler no grupo LeNet e de dataset no controle LR 2×, detalhada no mapa de experimentos. Uma comparação confirmatória deve congelar essas escolhas antes dos próximos testes.']
    (out/'convergence.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[6:16]))


if __name__=='__main__':
    main()
