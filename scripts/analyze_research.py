"""Reproduce the author's data investigation, emphasizing threshold diagnostics.

Run from any directory. Requires the plots and docs extras. Does not train models.
"""
import argparse
import base64
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from optimizer_research.diagnostics import (
    counter_diagnostics, pooled_window, threshold_bounds, curve_summary, first_sustained,
)

CURVES = {
    'Fashion-MNIST / LeNet': [
        ('AdamW', 'AdamW_losses.npy'), ('AdaSym', 'AdaSym_losses.npy'),
        ('AdaSym nored', 'AdaSym_loss_fmnist_nored.npy'),
        ('AdamW LR 2× [dataset ambíguo]', 'Adam_losses2x.npy')],
    'CIFAR-10 / ResNet-20': [
        ('AdamW', 'AdamW_losses_cifar10.npy'),
        ('AdaSymSigmoid', 'AdaSymSig_losses_cifar10.npy')],
}
COLORS = ['#087e8b', '#e19c37', '#64748b', '#a84f73']
BIN_LABELS = ['D ≤ v/4', 'v/4 < D ≤ v/2', 'D > v/2']


def save_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def save_figure(fig, out, name, plt):
    for extension in ('png', 'svg'):
        path = out/f'{name}.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight',
                    metadata={'Date': None} if extension == 'svg' else None)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)


def md_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def standalone_report(out):
    import markdown
    body = markdown.markdown((out/'report.md').read_text(encoding='utf-8'),
                             extensions=['tables', 'fenced_code'])
    # Inline figures make the reading copy portable and fully offline.
    for p in out.glob('*.png'):
        uri = 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode('ascii')
        body = body.replace(f'src="{p.name}"', f'src="{uri}"')
    style = '''body{max-width:1080px;margin:45px auto;padding:0 26px;color:#193544;
    font:17px/1.7 "Segoe UI",Arial,sans-serif;background:#fff}h1{font-size:36px;line-height:1.2}
    h2{margin-top:48px;border-bottom:2px solid #d8e9e7;padding-bottom:10px}h3{margin-top:32px}
    img{max-width:100%;height:auto}table{border-collapse:collapse;display:block;overflow-x:auto;
    font-size:14px;margin:24px 0}td,th{padding:10px 13px;border-bottom:1px solid #d5e0e3;
    text-align:left}th{background:#edf5f4}a{color:#087e8b}code{background:#eff3f4;padding:2px 4px}
    pre{padding:15px;background:#eff3f4;overflow:auto}blockquote{border-left:4px solid #087e8b;
    margin-left:0;padding-left:20px} @media print{h2,h3{break-after:avoid}img{break-inside:avoid}}
    '''
    (out/'report.html').write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>AdaSym — investigação dos dados</title><style>'+style+'</style><body>'+body+'</body></html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/original')
    parser.add_argument('--output', type=Path, default=ROOT/'results/research')
    args = parser.parse_args()
    data, out = args.data, args.output
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'adasym-research',
                         'axes.prop_cycle': matplotlib.cycler(color=COLORS)})

    arrays = {p.name: np.load(p, allow_pickle=False) for p in sorted(data.glob('*.npy'))}
    c4, c2, full = [arrays[f'AdaSym_fminist_{s}.npy'] for s in ('c4','c2','full')]
    d = counter_diagnostics(c4, c2, full)
    if len(full) != 150:
        raise ValueError('The author-confirmed experiment has 150 saved intervals')
    epochs = np.arange(1, len(full)+1)
    d4, d2, dt = d['increments']
    f4, middle, upper = d['pooled_rates']
    f2 = f4+middle
    records = []
    for i in range(len(full)):
        records.append(dict(epoch=int(i+1),c4_cumulative=int(c4[i]),c2_cumulative=int(c2[i]),
            full_cumulative=int(full[i]),c4_increment=int(d4[i]),c2_increment=int(d2[i]),
            full_increment=int(dt[i]),c4_rate=float(d['rates'][0,i]),
            between_v4_v2_rate=float(d['rates'][1,i]),above_v2_rate=float(d['rates'][2,i]),
            c2_rate=float(d['rates'][:2,i].sum()),
            c4_cumulative_rate=float(c4[i]/full[i]),c2_cumulative_rate=float(c2[i]/full[i])))
    save_csv(out/'threshold_by_epoch.csv', records)
    windows = [('1–10',0,10), ('11–40',10,40), ('41–70',40,70),
               ('71–100',70,100), ('101–130',100,130), ('131–150',130,150),
               ('141–150',140,150), ('1–150',0,150)]
    phases = []
    for label, start, stop in windows:
        b = pooled_window(d, start, stop)
        phases.append(dict(window=label,start_epoch=start+1,end_epoch=stop,
            evaluations=int(sum(dt[start:stop])),c4_rate=float(b[0]),
            middle_rate=float(b[1]),c2_rate=float(b[:2].sum()),above_v2_rate=float(b[2])))
    save_csv(out/'threshold_windows.csv', phases)
    bounds = threshold_bounds(f4, f2, [1,2,3,4,8,16])
    save_csv(out/'threshold_bounds.csv', bounds)

    fig, ax = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    ax[0,0].plot(epochs, d['rates'][0], label='≤ v/4: no piso')
    ax[0,0].plot(epochs, d['rates'][:2].sum(0), label='≤ v/2: inclui o piso')
    ax[0,0].set(title='Incidência em cada intervalo', ylabel='Fração das avaliações')
    ax[0,0].legend()
    ax[0,1].plot(epochs, c4/full, label='c4 / full')
    ax[0,1].plot(epochs, c2/full, label='c2 / full')
    ax[0,1].set(title='Razões acumuladas: suavizam mudanças recentes')
    ax[0,1].legend()
    ax[1,0].stackplot(epochs, d['rates'], labels=BIN_LABELS, colors=COLORS[:3])
    ax[1,0].set(title='Partição sem dupla contagem', ylabel='Fração das avaliações', ylim=(0,1))
    ax[1,0].legend(loc='center right', fontsize=9)
    ax[1,1].bar(epochs, np.asarray(dt,dtype=float)/1e9, width=1, color=COLORS[2])
    ax[1,1].set(title='Exposição observada por intervalo', ylabel='Bilhões de avaliações')
    for a in ax.flat:
        a.set_xlabel('Época / intervalo registrado')
        a.grid(alpha=.15)
    for a in (ax[0,0],ax[0,1],ax[1,0]):
        a.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('Diagnóstico do threshold · experimento de Ítalo Capuzzo\nLeNet / Fashion-MNIST · piso v/4 · 150 épocas', fontsize=14)
    save_figure(fig,out,'threshold_dynamics',plt)

    fig, ax = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    shown = phases[:6]
    positions = np.arange(len(shown))
    bottom = np.zeros(len(shown))
    for key, label, color in zip(('c4_rate','middle_rate','above_v2_rate'),BIN_LABELS,COLORS):
        vals = np.array([p[key] for p in shown])
        ax[0].bar(positions,vals,bottom=bottom,label=label,color=color)
        bottom += vals
    ax[0].set(xticks=positions,xticklabels=[p['window'] for p in shown],
              xlabel='Janela de épocas', ylabel='Fração ponderada por avaliações',
              title='Exposição por fase do treinamento',ylim=(0,1))
    ax[0].legend(fontsize=8)
    ax[0].yaxis.set_major_formatter(PercentFormatter(1))
    rolling = np.stack([pooled_window(d,max(0,i-9),i+1) for i in range(len(full))],axis=1)
    ax[1].plot(epochs,rolling[0],label='≤ v/4')
    ax[1].plot(epochs,rolling[:2].sum(0),label='≤ v/2')
    ax[1].set(title='Janela móvel: até 10 intervalos',xlabel='Última época da janela',
              ylabel='Contagens somadas / exposição somada')
    ax[1].yaxis.set_major_formatter(PercentFormatter(1))
    ax[1].legend()
    fig.suptitle('O comportamento do limitador muda ao longo do treino',fontsize=14)
    save_figure(fig,out,'threshold_phases',plt)

    fig, ax = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    vals = [f4,middle,upper]
    bars = ax[0].barh(BIN_LABELS,vals,color=COLORS[:3])
    ax[0].bar_label(bars,labels=[f'{v:.2%}' for v in vals],padding=5)
    ax[0].set(xlim=(0,1),title='Distribuição identificada pelos contadores',xlabel='Fração das avaliações')
    ax[0].xaxis.set_major_formatter(PercentFormatter(1))
    for i,b in enumerate(bounds):
        ax[1].plot([i,i],[b['lower'],b['upper']],color=COLORS[2],linewidth=4)
        if b['identified']:
            ax[1].scatter(i,b['lower'],color=COLORS[0],s=65,zorder=4)
        else:
            ax[1].scatter([i,i],[b['lower'],b['upper']],color=COLORS[2],marker='_',s=140)
    ax[1].set(xticks=range(len(bounds)),xticklabels=[f"v/{b['divisor']:g}" for b in bounds],
        ylim=(-.02,1.03),xlabel='Piso considerado na mesma trajetória',ylabel='Fração compatível com os dados',
        title='Sensibilidade: pontos exatos e limites identificáveis')
    ax[1].yaxis.set_major_formatter(PercentFormatter(1))
    ax[1].text(.5,.94,'Barras = limites; não são intervalos de confiança',transform=ax[1].transAxes,
               ha='center',va='top',fontsize=8)
    fig.suptitle('Dois thresholds observados delimitam a análise de sensibilidade',fontsize=14)
    save_figure(fig,out,'threshold_sensitivity',plt)

    metrics, targets = [], []
    fig, ax = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for col,(group,methods) in enumerate(CURVES.items()):
        for j,(label,filename) in enumerate(methods):
            a = arrays[filename]
            if a.shape != (3,150) or not np.isfinite(a).all():
                raise ValueError(f'Unexpected curves: {filename}')
            for row,curve in enumerate(a):
                metrics.append(dict(group=group,method=label,file=filename,row=row,
                                    **curve_summary(curve)))
                for target in (.05,.1,.2):
                    for width in (1,5,10):
                        reached = first_sustained(curve,target,width)
                        targets.append(dict(group=group,method=label,file=filename,row=row,
                            target=target,width=width,epoch=reached,reached=reached is not None,horizon=150))
            mean, sd = a.mean(0), a.std(0,ddof=1)
            color=COLORS[j]
            ax[0,col].plot(epochs,mean,label=label,color=color)
            ax[0,col].fill_between(epochs,np.maximum(mean-sd,np.finfo(float).tiny),mean+sd,color=color,alpha=.12)
            # Difference of group means; row identities are not paired.
            if j:
                base = arrays[methods[0][1]].mean(0)
                ax[1,col].plot(epochs,(base-mean)/base,label=label,color=color)
        ax[0,col].set(title=group,yscale='log',ylabel='Loss registrada · média ± DP entre trajetórias')
        ax[0,col].legend(fontsize=8)
        ax[1,col].axhline(0,color=COLORS[2],lw=.7)
        ax[1,col].set(xlabel='Época registrada',ylabel='(AdamW − variante) / AdamW')
        ax[1,col].yaxis.set_major_formatter(PercentFormatter(1))
        ax[1,col].legend(fontsize=8)
    fig.suptitle('Loss e contraste com AdamW · descrição dos experimentos do autor\nFaixas = dispersão entre 3 trajetórias; não são ICs',fontsize=14)
    save_figure(fig,out,'loss_diagnostics',plt)
    save_csv(out/'loss_by_trajectory.csv',metrics)
    save_csv(out/'convergence_targets.csv',targets)
    # Keep the 3×100 file as three methods, never three repetitions.
    fashion100 = [dict(method=label,row=i,**curve_summary(arrays['losses_FashionMNIST.npy'][i]))
                  for i,label in enumerate(('AdaSym','AdaSymSigmoid','AdamW'))]
    save_csv(out/'fashion100_methods.csv',fashion100)

    lr = np.concatenate([np.arange(1,10)*scale for scale in (1e-4,1e-3,1e-2)])
    lr_a,lr_w = arrays['lrs_aucloss.npy'],arrays['lrs_aucloss_adamW.npy']
    if lr_a.shape != (27,) or lr_w.shape != (27,):
        raise ValueError('Expected 27-point LR vectors')
    save_csv(out/'learning_rate_scan.csv',[dict(index=i,lr=float(x),adasym_context_inferred=float(a),
        adamw=float(w),relative_difference=float((w-a)/w)) for i,(x,a,w) in enumerate(zip(lr,lr_a,lr_w))])
    grids = []
    fig, ax = plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    ax[0].plot(lr,lr_a,'o-',label='AdaSym [contexto do metadata]')
    ax[0].plot(lr,lr_w,'o-',label='AdamW')
    ax[0].set(xscale='log',yscale='log',xlabel='LR inferida do metadata',
              ylabel='Objetivo salvo [AUC no metadata]',title='27 pontos · comparação de LR')
    ax[0].legend(fontsize=7)
    for axis,(filename,e3,e4) in zip(ax[1:],[('loss_matrix.npy',(-.3,0),(-3,0)),
                    ('loss_matrix (1).npy',(-.3,-.01),(-1.5,0))]):
        a = arrays[filename]
        b3,b4 = 10**np.linspace(*e3,10),10**np.linspace(*e4,70)
        finite = np.isfinite(a)
        masked = np.where(finite,a,np.nan)
        idx=np.unravel_index(np.nanargmin(masked),a.shape)
        for row,col in np.ndindex(a.shape):
            grids.append(dict(file=filename,row=row,col=col,beta3=float(b3[row]),beta4=float(b4[col]),
                value=float(a[row,col]) if finite[row,col] else None,finite=bool(finite[row,col])))
        image=axis.pcolormesh(b4,b3,np.ma.masked_invalid(masked),shading='nearest',cmap='viridis',rasterized=True)
        axis.scatter(b4[idx[1]],b3[idx[0]],marker='*',s=140,color='#ffffff',edgecolor='#172b36')
        axis.set(xscale='log',xlabel='beta4',ylabel='beta3',title=filename)
        fig.colorbar(image,ax=axis,label='Objetivo salvo')
    fig.suptitle('Exploração dos hiperparâmetros · ótimos observados, sem garantia de transferência',fontsize=14)
    save_figure(fig,out,'hyperparameter_diagnostics',plt)
    save_csv(out/'beta_grid_cells.csv',grids)

    # Every quantitative statement below is calculated from the supplied arrays.
    first,last = phases[0],phases[-2]
    loss_rows = []
    for group,methods in CURVES.items():
        for label,filename in methods:
            rows=[r for r in metrics if r['file']==filename]
            auc=np.mean([r['normalized_auc'] for r in rows])
            late=np.mean([r['last_window_mean'] for r in rows])
            spread=np.std([r['last_window_mean'] for r in rows],ddof=1)
            temporal=np.mean([r['temporal_sd'] for r in rows])
            residual=np.mean([r['detrended_rms'] for r in rows])
            loss_rows.append([group,label,f'{auc:.6f}',f'{late:.6f} ± {spread:.6f}',f'{temporal:.6f}',f'{residual:.6f}'])
    lines = [
        '# AdaSym: investigação experimental orientada por dados', '',
        '**Pesquisa de Ítalo Capuzzo · diagnóstico do otimizador, sensibilidade do threshold e evolução da fórmula.**', '',
        'O AdaSym é uma pesquisa experimental em algoritmos de otimização para machine learning. '
        'Ítalo desenvolveu os experimentos, analisou curvas e estatísticas internas, identificou padrões '
        'e revisou hipóteses sobre a regra de atualização. Ciência de dados faz parte dessa metodologia: '
        'as visualizações participam da investigação e da evolução da fórmula, além da comunicação dos resultados. '
        'Os números a seguir quantificam os registros dessa pesquisa; não representam novos treinamentos.', '',
        '## 1. Perguntas que orientam o diagnóstico', '',
        '| Pergunta | Evidência examinada | O que se pode responder |',
        '|---|---|---|',
        '| O piso domina a atualização ou atua em uma parcela das avaliações? | c4 / full | Incidência do limite v/4 |',
        '| Quanto muda a incidência ao considerar v/2? | c2 − c4 e c2 / full | Sensibilidade observada a dois thresholds na mesma trajetória |',
        '| A incidência permanece constante ao longo do treino? | Diferenças dos contadores acumulados | Mudança temporal por intervalo e fase |',
        '| Menor loss inicial implica melhor loss final? | Curvas, AUC, trecho final e controle LR 2× | Compromissos entre velocidade e qualidade tardia |',
        '| Um mínimo isolado basta para escolher uma fórmula? | Grades de betas e LR | Configurações promissoras e regiões inválidas, com limites de seleção |', '',
        '## 2. A evolução da fórmula é parte do raciocínio científico do autor', '',
        'A apresentação do autor registra uma sequência de regras: expressão descoberta, quadrado da memória, '
        'piso relativo, memória separada e peso temporal. Segundo o relato de Ítalo, a análise dos dados e a '
        'identificação de padrões participaram das conclusões que conduziram a essa evolução. '
        'Os slides e arrays documentam a sequência e os testes; não oferecem um diário que associe cada '
        'alteração a uma única execução específica.', '',
        md_table(['Etapa documentada','Regra do denominador','Questão investigada'],[
            ['Expressão inicial','sqrt(abs(v − m1))','Explorar uma regra candidata encontrada na busca simbólica'],
            ['Quadrado da memória','sqrt(abs(v − m1²))','Relacionar a memória à escala de segundo momento'],
            ['Piso relativo','sqrt(max(v − m1², v/c))','Limitar a amplificação e investigar frequência de saturação'],
            ['Memória separada','sqrt(max(v − m2², v/c))','Investigar coeficientes próprios no denominador'],
            ['Peso temporal','sqrt(max(v − s(e)·m2², v/c))','Investigar a duração da correção de consistência']]), '',
        'Essa leitura conecta formulação, observação e revisão de hipóteses. O diagnóstico dos contadores '
        'mostra quando o mecanismo de proteção participa da atualização; as comparações de loss e LR '
        'investigam o resultado do treinamento. Esses dois níveis de evidência se complementam, mas '
        'frequência de saturação sozinha não determina qual fórmula tem melhor desempenho.', '',
        '## 3. Threshold: definição, unidade e integridade', '',
        'Considere D = v − s·m2² antes do piso e d = max(D, v/4). '
        'c4 conta avaliações com denominador ≤ v/4; c2 conta avaliações ≤ v/2; '
        'full conta o total de avaliações de gradientes/parâmetros. '
        '**São avaliações repetidas de coordenadas ao longo do treino, não parâmetros distintos, '
        'passos do otimizador ou réplicas estatisticamente independentes.**', '',
        'O autor confirmou que “1550 epochs” no metadata significa **150 épocas**. '
        'A correção está em `data/author_clarifications.json`; o arquivo original foi preservado. '
        'Os contadores são inteiros, não negativos, crescentes e aninhados. '
        'A propriedade c4 ≤ c2 ≤ full também vale para os incrementos de cada intervalo.', '',
        'Como os registros são acumulados, para a posição e usamos ΔC[e] = C[e] − C[e−1], '
        'com C[0] = 0. A taxa por intervalo é Δc4 / Δfull, e não c4 / full. '
        'Para janelas, somamos os numeradores e denominadores antes de dividir. '
        'Isso dá o peso correto à exposição de cada intervalo.', '',
        f'O total registrado é **{int(full[-1]):,} avaliações**. O primeiro intervalo contém '
        f'{int(dt[0]):,}, enquanto o segundo contém {int(dt[1]):,} '
        f'({float(dt[0]/dt[1]):.2%} da exposição de um intervalo posterior). '
        'A diferença de exposição é observada; sua causa não está documentada. '
        'Por isso, a primeira posição não é tratada automaticamente como uma época completa.', '',
        '## 4. Quanto o limitador atua?', '',
        md_table(['Evento disjunto','Avaliações','Fração do total'],[
            ['D ≤ v/4',f'{int(c4[-1]):,}',f'{f4:.4%}'],
            ['v/4 < D ≤ v/2',f'{int(c2[-1])-int(c4[-1]):,}',f'{middle:.4%}'],
            ['D > v/2',f'{int(full[-1])-int(c2[-1]):,}',f'{upper:.4%}']]), '',
        f'No histórico do autor, o limite v/4 foi alcançado em **{f4:.2%}** das avaliações. '
        f'O conjunto até v/2 representa **{f2:.2%}**. A faixa intermediária soma '
        f'**{middle*100:.2f} pontos percentuais** e corresponde a um aumento de '
        f'**{middle/f4:.2%}** sobre a incidência em v/4. As categorias c4 e c2 são aninhadas: '
        'somá-las diretamente produziria dupla contagem.', '',
        f'Esse padrão é incompatível com a descrição de que o piso v/4 domina todas ou quase todas '
        f'as avaliações desse treinamento: **{1-f4:.2%}** ficam acima desse limite. '
        'Ao mesmo tempo, a incidência observada mostra que a proteção participa de uma parcela '
        'mensurável das atualizações. Ela não prova que o piso seja dispensável, nem mede '
        'quanto essas coordenadas contribuem para a norma do update ou para a redução da loss.', '',
        '![Dinâmica dos contadores](threshold_dynamics.png)', '',
        '## 5. A sensibilidade varia ao longo do treinamento?', '',
        md_table(['Janela','≤ v/4','v/4 < D ≤ v/2','≤ v/2'],[
            [p['window'],f"{p['c4_rate']:.3%}",f"{p['middle_rate']:.3%}",f"{p['c2_rate']:.3%}"] for p in phases]), '',
        f'Na janela 1–10, a incidência em v/4 é **{first["c4_rate"]:.2%}**; '
        f'na janela 141–150, é **{last["c4_rate"]:.2%}**. A diferença é '
        f'**{(last["c4_rate"]-first["c4_rate"])*100:.2f} pontos percentuais** '
        f'({(last["c4_rate"]/first["c4_rate"]-1):.2%} em termos relativos). '
        'A comparação descreve duas janelas escolhidas para exploração; não pressupõe uma queda '
        'monótona em todas as épocas nem define pontos de mudança estatisticamente estimados.', '',
        'O acompanhamento por intervalo permite revisar a hipótese de incidência constante. '
        'As razões acumuladas retêm a influência do início e podem esconder mudanças recentes. '
        'As janelas móveis usam até dez intervalos e são ponderadas pela exposição, sem alterar os contadores.', '',
        '![Threshold por fase](threshold_phases.png)', '',
        '## 6. Sensibilidade identificável e o que exige outra execução', '',
        'Os dois contadores fornecem dois pontos observados da distribuição do denominador em relação a v. '
        'Para v > 0, são pontos da distribuição empírica de D/v em 1/4 e 1/2. '
        'A presença de v = 0 não foi registrada; por isso a análise usa os eventos relativos '
        'D ≤ v/k, sem criar razões individuais D/v que não estão nos arquivos.', '',
        md_table(['Piso considerado','Limite inferior de incidência','Limite superior','Identificação'],[
            [f"v/{b['divisor']:g}",f"{b['lower']:.3%}",f"{b['upper']:.3%}",
             'Ponto observado' if b['identified'] else 'Somente limites'] for b in bounds]), '',
        'Esses limites vêm apenas da ordem dos eventos, sem interpolar uma densidade não observada. '
        'Eles se referem à mesma trajetória e não estimam como seria um novo treino com outro piso. '
        'Alterar o threshold muda os parâmetros, os gradientes futuros e os próprios momentos. '
        'As barras do gráfico são limites de identificação, não intervalos de confiança.', '',
        'Para v > 0, mesmos momentos e epsilon desprezível, o piso v/4 limita o ganho da parte '
        'adaptativa a 2× em relação à normalização por sqrt(v). A região até v/4 fica no teto; '
        'a faixa v/4 < D ≤ v/2 corresponde a ganhos entre sqrt(2) e 2. '
        'Isso explica por que a investigação de Ítalo sobre saturação informa a interpretação '
        'da fórmula. A frequência dos eventos não recupera a distribuição completa dos ganhos '
        'nem seus efeitos causais na loss.', '',
        '![Limites da sensibilidade](threshold_sensitivity.png)', '',
        '## 7. A loss responde a outra parte da hipótese', '',
        'As matrizes 3 × 150 são resumidas por trajetória e por grupo. '
        'Não se presume pareamento entre linhas de arquivos diferentes. '
        'O arquivo 3 × 100 contém três métodos distintos, conforme a ordem inferida dos slides, '
        'e é exportado separadamente. AUC normalizada é a integral trapezoidal entre as posições '
        '1 e N dividida por N−1; não é a média aritmética das N losses nem a AUC histórica das varreduras.', '',
        md_table(['Grupo','Método','AUC normalizada','Últimas 20: média ± DP entre trajetórias',
                  'DP temporal médio','RMS após remover tendência linear'],loss_rows), '',
        'O DP entre trajetórias descreve a variação de seus resumos finais. '
        'O DP temporal descreve a variação dentro da janela de cada trajetória e inclui tendência. '
        'O RMS residual remove apenas uma tendência linear. Nenhuma dessas medidas demonstra '
        'estabilidade teórica do otimizador. Sementes e split da loss não estão identificados.', '',
        'O controle com LR 2× tem melhor resumo da curva inteira que AdaSym, enquanto AdaSym '
        'tem loss tardia menor nesse grupo de arquivos. A variante nored expõe outro compromisso: '
        'loss tardia baixa com maior variação temporal. Assim, a análise dos experimentos do autor '
        'examina o que significa “melhor”: acelerar o início, reduzir a loss final ou reduzir a oscilação. '
        'O conflito de rótulo do dataset no controle LR 2× permanece documentado.', '',
        '![Diagnóstico das losses](loss_diagnostics.png)', '',
        'O arquivo `convergence_targets.csv` mantém o resultado de cada trajetória para os alvos '
        '0,05 / 0,1 / 0,2, exigindo 1 / 5 / 10 posições consecutivas. Campos de época vazios '
        'significam alvo não alcançado dentro do horizonte; nunca são substituídos por zero. '
        'Essas escolhas são exploratórias e permitem verificar a sensibilidade da descrição ao critério.', '',
        '## 8. Exploração dos hiperparâmetros', '',
        f'Na grade de 27 LRs reconstruída do metadata, o mínimo do vetor associado a AdaSym '
        f'é {float(lr_a.min()):.6f} em LR {float(lr[lr_a.argmin()]):g}; '
        f'o mínimo de AdamW é {float(lr_w.min()):.6f} em LR {float(lr[lr_w.argmin()]):g}. '
        'Comparar métodos na mesma LR e comparar seus mínimos ajustados são perguntas diferentes. '
        'O significado AUC vem do metadata; o slide correspondente descreve outro resumo. '
        'A figura mostra o objetivo salvo sem resolver essa divergência.', '',
        'As duas grades beta3/beta4 expõem a estrutura do objetivo e os valores inválidos. '
        'A grade `loss_matrix.npy` tem 70 valores não finitos, preservados como ausentes. '
        'Os mínimos são observações selecionadas na própria grade, sem repetições identificadas '
        'ou estimativa de incerteza. O vetor de 80 posições não recebe um eixo de LR inventado; '
        'o vetor de transição da sigmoide possui índices, mas endpoints exatos desconhecidos.', '',
        '![Exploração dos hiperparâmetros](hyperparameter_diagnostics.png)', '',
        '## 9. Conclusões e hipóteses que os dados permitem revisar', '',
        '- **O limitador atua de forma seletiva neste histórico.** A incidência de v/4 é próxima '
        'de 10%, o que afasta a hipótese de saturação quase universal nessa execução.',
        '- **O comportamento varia com a fase do treino.** Separar incrementos e acumulados '
        'revela mudanças temporais que um único percentual final não descreve.',
        '- **O threshold altera a população atingida.** O evento até v/2 inclui uma faixa '
        'intermediária mensurável; dois pontos não identificam uma curva contínua de sensibilidade.',
        '- **O ganho observado depende da métrica e do controle.** Curva inteira, loss tardia '
        'e LR maior precisam acompanhar qualquer interpretação sobre a fórmula.',
        '- **A evolução da regra expressa uma investigação do autor.** Ítalo combinou '
        'busca de expressões, diagnóstico dos mecanismos e comparação experimental para '
        'desenvolver o AdaSym. As figuras documentam perguntas e padrões examinados nessa pesquisa.', '',
        'Não há evidência para atribuir essas taxas a camadas específicas, neurônios, sinais dos gradientes '
        'ou épocas de uma curva de loss particular: os contadores são globais e não identificam '
        'a trajetória correspondente. Também não é possível calcular a contribuição para a norma do update, '
        'um efeito causal na generalização ou um “melhor threshold” a partir dessas contagens.', '',
        '## 10. Continuação da investigação', '',
        'Uma próxima rodada pode registrar D/v (se v > 0), ganho aplicado, norma do update e '
        'taxas por camada e por época, incluindo as avaliações com v = 0 em categoria própria. '
        'Comparar c = 2 / 4 / 8, com orçamento equivalente de ajuste de LR, permite relacionar '
        'saturação e resultado do treino. Registrar seeds, splits e IDs de execução vincula '
        'cada diagnóstico à sua loss e acurácia. Essa rodada deve separar a seleção de '
        'hipóteses dos dados usados para confirmá-las.', '',
        'Não são calculados p-valores ou ICs binomiais sobre trilhões de avaliações correlacionadas. '
        'Frequências de coordenadas no mesmo treinamento não constituem trilhões de réplicas. '
        'Réplicas independentes e instrumentação completa são necessárias para quantificar '
        'incerteza entre treinamentos e testar efeitos da escolha do piso.', '',
        '## Reprodução e tabelas', '',
        'Na raiz do repositório:', '',
        '```bash\npython -m pip install -e ".[plots,docs]"\npython scripts/analyze_research.py\npython -m unittest discover -s tests -v\n```', '',
        'As tabelas CSV mantêm contagens inteiras, métricas por trajetória, alvos não alcançados, '
        'grades e limites de sensibilidade. `analysis_manifest.json` registra hashes dos dados, '
        'parâmetros da análise, arquivos-fonte e versões do ambiente. '
        '`report.html` incorpora todas as figuras e pode ser aberto sem internet. '
        'As figuras são exportadas em PNG e SVG.',
    ]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    source_files = [Path(__file__), ROOT/'src/optimizer_research/diagnostics.py',
                    ROOT/'data/author_clarifications.json']
    manifest = dict(author='Ítalo Capuzzo',experiment='historical author experiments; no new training',
        counter_unit='repeated coordinate evaluations',counter_epochs=150,
        cumulative=True,zero_initial_counts_assumed=True,
        targets=[.05,.1,.2],sustained_widths=[1,5,10],late_window=20,
        rolling_window=10,inference='descriptive; no iid coordinates or paired seeds assumed',
        environment=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
        inputs=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in sorted(data.iterdir()) if p.suffix in ('.npy','.txt')],
        analysis_sources=[dict(file=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for p in source_files],
        pooled_threshold_rates=dict(c4=float(f4),c2=float(f2),middle=float(middle),above_v2=float(upper)))
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    standalone_report(out)
    print(json.dumps(dict(output=str(out),threshold_v4=float(f4),threshold_v2=float(f2),
                          first10=float(first['c4_rate']),last10=float(last['c4_rate'])),ensure_ascii=False))


if __name__ == '__main__':
    main()
