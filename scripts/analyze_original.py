"""Audit immutable historical arrays without guessing seeds or row labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=root/'data/original')
    parser.add_argument('--output', type=Path, default=root/'results/historical')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory, curves = [], []
    arrays = {}
    for path in sorted(args.data.glob('*.npy')):
        a = np.load(path, allow_pickle=False)
        arrays[path.name] = a
        finite = np.isfinite(a)
        values = a[finite]
        inventory.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=list(a.shape), dtype=str(a.dtype), nan=int(np.isnan(a).sum()), inf=int(np.isinf(a).sum()), finite_min=float(values.min()) if values.size else None, finite_max=float(values.max()) if values.size else None))
        # Mixed-optimizer file is deliberately excluded from row aggregation.
        if a.ndim == 2 and a.shape == (3,150) and 'loss' in path.name.lower():
            for i, row in enumerate(a):
                curves.append(dict(file=path.name, row=i, first=float(row[0]), last=float(row[-1]), mean_last_10=float(row[-10:].mean()), mean_over_epochs=float(row.mean()), trapezoid_auc=float(np.sum((row[:-1]+row[1:])/2)), interpretation='trajectory; seed and split unknown'))
    (args.output/'inventory.json').write_text(json.dumps(inventory, indent=2, allow_nan=False), encoding='utf-8')
    with (args.output/'curves.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0]))
        writer.writeheader(); writer.writerows(curves)
    lines = ['# Auditoria dos resultados originais', '', 'Cada linha das matrizes 3 × 150 é tratada como trajetória sem identidade confirmada. A tabela agrega essas três trajetórias descritivamente. Não se presume que sejam sementes independentes ou que a loss seja de validação. Arquivos e hashes constam em `inventory.json`. AUC usa espaçamento de uma época, começando na primeira loss registrada. O código do cálculo original de AUC não foi fornecido.', '', '| Arquivo | Média das últimas 10 posições | Média de todas as posições |', '|---|---:|---:|']
    for name in sorted({r['file'] for r in curves}):
        rows = [r for r in curves if r['file'] == name]
        lines.append(f"| {name} | {np.mean([r['mean_last_10'] for r in rows]):.6f} | {np.mean([r['mean_over_epochs'] for r in rows]):.6f} |")
    lines += ['', '## Cruzamento com o PowerPoint', '', 'Os gráficos dos slides 10, 12 e 14 identificam médias e desvios padrão, coerentes com três trajetórias por arquivo 3 × 150. Isso sustenta a interpretação como repetições, mas não identifica sementes ou independência. O gráfico do slide 16 explicita contadores acumulados, reforçando essa leitura dos arquivos c2/c4/full.', '', 'Para `losses_FashionMNIST.npy` (3 × 100), a correspondência dos valores iniciais e finais com os gráficos dos slides 8 e 9 permite inferir a ordem abaixo. É uma inferência visual verificável, ainda sem o programa que gerou o gráfico. Essas linhas representam métodos diferentes e não três sementes do mesmo método.', '', '| Linha (índice zero) | Método inferido | Primeiro valor | Último valor | Média registrada |', '|---|---|---:|---:|---:|']
    for index, label in enumerate(('AdaSym','AdaSymSigmoid','AdamW')):
        row = arrays['losses_FashionMNIST.npy'][index]
        lines.append(f'| {index} | {label} | {row[0]:.6f} | {row[-1]:.6f} | {row.mean():.6f} |')
    lines += ['', 'Os gráficos dos slides 18 e 19 indicam mínimos em (0,5843; 0,7406) e (0,6744; 0,4062). A reconstrução dos eixos pelo metadata e os mínimos numéricos abaixo coincidem com essas marcações. Essa concordância identifica as duas varreduras, sem resolver o significado de beta4.', '', '## Inconsistências e limites', '', '- `loss_matrix.npy` inclui betas iguais a 1 na grade declarada, inválidos para uma EMA normalizada com bias correction. A presença de NaN coincide com a última linha, mas falta código para estabelecer a causa.', '- `lrs_AdaSym_aucloss.npy` tem 80 valores. A grade com 27 LRs em metadata não permite reconstruir seu eixo. Não se atribui uma LR ótima a esse arquivo.', '- O slide 17 rotula sua métrica como média das últimas 15 losses e mostra LRs até 1e-2. Os vetores de LR em metadata são descritos como AUC e incluem valores até 9e-2. Não se presume que sejam a fonte exata desse gráfico sem o pós-processamento original.', '- `metadata.txt` diz 1550 épocas para os contadores. Os três arquivos correspondentes têm 150 posições, compatíveis com o eixo do slide 16. Preserva-se a divergência textual no registro original.', '- `Adam_losses2x.npy` é chamado AdamW 2× no texto e nos gráficos, mas o nome diz Adam. O dataset aparece como MNIST em metadata e Fashion-MNIST nos slides. A comparação não é rotulada como controlada.', '- A sigmoide usa ponto de transição 5 no exemplo visual e 15 em metadata. As recorrências de m2, bias correction, epsilon e scheduler completo estão ausentes.', '- Ausência de acurácia, separação treino/validação/teste, sementes, checkpoints, tempos, versões e logs da busca impede concluir superioridade de generalização ou reconstruir a regressão simbólica histórica.', '', '## Grade de betas', '', '| Arquivo | Não finitos | Menor valor finito | Índice mínimo | Betas inferidos por metadata |', '|---|---:|---:|---|---|']
    for name, exp3, exp4 in [('loss_matrix.npy',(-.3,0),(-3,0)), ('loss_matrix (1).npy',(-.3,-.01),(-1.5,0))]:
        a = arrays[name]
        index = tuple(int(i) for i in np.unravel_index(np.nanargmin(np.where(np.isfinite(a), a, np.nan)), a.shape))
        b3 = 10**np.linspace(*exp3,10)[index[0]]
        b4 = 10**np.linspace(*exp4,70)[index[1]]
        lines.append(f'| {name} | {np.size(a)-np.isfinite(a).sum()} | {a[index]:.6f} | {index} | ({b3:.6f}, {b4:.6f}) |')
    lines += ['', 'Esses mínimos são observações de uma varredura, sem repetições identificadas. Não constituem hiperparâmetros confirmados. O papel de beta4 não é definido e não equivale automaticamente ao complemento 1−beta3.', '', '## Contadores', '']
    full = arrays['AdaSym_fminist_full.npy']
    for suffix in ('c2','c4'):
        count = arrays[f'AdaSym_fminist_{suffix}.npy']
        lines.append(f'- `{suffix}`: razão acumulada final = {count[-1]/full[-1]:.4%}. Se os arquivos registram contadores acumulados compatíveis, a razão por época deve usar diferenças sucessivas do numerador e denominador, não a razão acumulada em cada época.')
    lines += ['', 'As curvas registradas sugerem diferenças na dinâmica de treinamento. Os dados disponíveis não sustentam a afirmação de que um otimizador generaliza melhor nem separam o efeito da janela do efeito de uma LR maior.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Audit complete; install .[plots] to regenerate figures.')
        return
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(1,2,figsize=(12,4), constrained_layout=True)
    for file, label in [('AdamW_losses.npy','AdamW'),('AdaSym_losses.npy','AdaSym (arquivo)'),('Adam_losses2x.npy','Adam/AdamW 2x (ambíguo)')]:
        a = arrays[file]
        ax[0].plot(np.arange(1,151), a.mean(axis=0), label=label)
    for file,label in [('AdamW_losses_cifar10.npy','AdamW'),('AdaSymSig_losses_cifar10.npy','AdaSymSig')]:
        ax[1].plot(np.arange(1,151),arrays[file].mean(axis=0),label=label)
    for axis,title in zip(ax,['Fashion-MNIST (rótulo parcial)','CIFAR-10 (rótulo dos arquivos)']):
        axis.set(xlabel='Posição registrada / época presumida',ylabel='Loss registrada',title=title,yscale='log')
        axis.legend(fontsize=8)
    fig.suptitle('Histórico: médias de 3 trajetórias; split e sementes desconhecidos', fontsize=11)
    fig.savefig(args.output/'historical_curves.png',dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for axis,name in zip(ax,['loss_matrix.npy','loss_matrix (1).npy']):
        a = arrays[name]
        im = axis.imshow(np.ma.masked_invalid(a),aspect='auto',origin='lower',cmap='viridis')
        axis.set(xlabel='Índice beta4',ylabel='Índice beta3',title=name)
        fig.colorbar(im,ax=axis,label='Valor registrado')
    fig.suptitle('Grades históricas: células ausentes indicam valores inválidos')
    fig.savefig(args.output/'beta_grids.png',dpi=180)
    plt.close(fig)
    print('\n'.join(lines[:16]))


if __name__ == '__main__':
    main()
