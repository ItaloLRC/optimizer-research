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
    boost_diagnostics, curve_summary, first_sustained,
)

CURVES = {
    'Fashion-MNIST / LeNet': [
        ('AdamW', 'AdamW_losses.npy'), ('AdaSym', 'AdaSym_losses.npy'),
        ('AdaSym nored', 'AdaSym_loss_fmnist_nored.npy'),
        ('Controle LR 2×', 'Adam_losses2x.npy')],
    'CIFAR-10 / ResNet-20': [
        ('AdamW', 'AdamW_losses_cifar10.npy'),
        ('AdaSymSigmoid', 'AdaSymSig_losses_cifar10.npy')],
}
COLORS = ['#087e8b', '#e19c37', '#64748b', '#a84f73']


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

    d = boost_diagnostics(c4,c2,full)
    epochs = np.arange(1,len(full)+1)
    d4,d2,df = d['increments']
    records = [dict(epoch=i+1,c4_cumulative=int(c4[i]),c2_cumulative=int(c2[i]),
        full_cumulative=int(full[i]),c4_increment=int(d4[i]),c2_increment=int(d2[i]),
        full_increment=int(df[i]),c2_outside_c4_increment=int(d2[i]-d4[i]),
        c4_within_c2=float(d4[i]/d2[i]) if d2[i] else None,
        c4_within_c2_cumulative=float(c4[i]/c2[i]) if c2[i] else None)
        for i in range(len(full))]
    save_csv(out/'threshold_by_epoch.csv',records)
    windows = [('1–10',0,10),('11–40',10,40),('41–70',40,70),
               ('71–100',70,100),('101–130',100,130),('131–150',130,150)]
    phases=[]
    for label,start,stop in windows:
        a,b,f=[int(sum(x[start:stop])) for x in (d4,d2,df)]
        phases.append(dict(window=label,c4=a,c2=b,full=f,c2_outside_c4=b-a,
                           c4_within_c2=a/b if b else None))
    save_csv(out/'threshold_windows.csv',phases)
    totals=[int(x[-1]) for x in (c4,c2,full)]
    save_csv(out/'boost_counter_summary.csv',[
        dict(counter=name,total=value) for name,value in zip(('c4','c2','full'),totals)])

    fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for name,a,delta,color in zip(('c4','c2','full'),(c4,c2,full),(d4,d2,df),COLORS):
        ax[0,0].plot(epochs,a/1e9,label=name,color=color)
        ax[0,1].plot(epochs,np.asarray(delta,dtype=float)/1e9,label=name,color=color)
    ax[0,0].set(title='Contadores acumulados',ylabel='Bilhões de registros',yscale='log')
    ax[0,1].set(title='Incrementos por intervalo',ylabel='Bilhões de registros')
    ax[1,0].stackplot(epochs,d['partial_counts']/1e9,labels=['c4','c2 fora de c4'],colors=COLORS[:2])
    ax[1,0].set(title='Composição das contagens parciais de c2',ylabel='Bilhões de registros por intervalo')
    ax[1,1].plot(epochs,d['within_c2'][0],label='c4 / c2 por intervalo',color=COLORS[0])
    ax[1,1].plot(epochs,c4/c2,label='c4 / c2 acumulado',color=COLORS[1])
    ax[1,1].set(title='Participação de c4 dentro de c2',ylabel='Fração dos registros de c2')
    ax[1,1].yaxis.set_major_formatter(PercentFormatter(1))
    for a in ax.flat:
        a.set_xlabel('Época / intervalo registrado')
        a.legend(fontsize=9)
        a.grid(alpha=.15)
    fig.suptitle('Diagnóstico do boost direcional · LeNet / Fashion-MNIST · 150 épocas',fontsize=14)
    save_figure(fig,out,'threshold_dynamics',plt)

    fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(len(phases))
    within=np.array([v['c4_within_c2'] for v in phases])
    ax[0].bar(x,within,label='c4',color=COLORS[0])
    ax[0].bar(x,1-within,bottom=within,label='c2 fora de c4',color=COLORS[1])
    ax[0].set(xticks=x,xticklabels=[v['window'] for v in phases],ylim=(0,1),
        title='Composição interna de c2 por fase',ylabel='Fração dos registros de c2',xlabel='Janela de épocas')
    ax[0].yaxis.set_major_formatter(PercentFormatter(1))
    rolling=np.array([float(sum(d4[max(0,i-9):i+1])/sum(d2[max(0,i-9):i+1]))
                      for i in range(len(full))])
    ax[1].plot(epochs,rolling,color=COLORS[0],label='c4 / c2 · até 10 intervalos')
    ax[1].set(title='Evolução da composição parcial',xlabel='Último intervalo da janela',ylabel='Fração dos registros de c2')
    ax[1].yaxis.set_major_formatter(PercentFormatter(1))
    for a in ax: a.legend(fontsize=8)
    save_figure(fig,out,'threshold_phases',plt)

    fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    labels=['c4','c2 fora de c4']
    values=[totals[0],totals[1]-totals[0]]
    bars=ax[0].bar(labels,np.array(values)/1e9,color=COLORS[:2])
    ax[0].bar_label(bars,labels=[f'{v:,}' for v in values],padding=4)
    ax[0].set(title='Contagens parciais: partição de c2',ylabel='Bilhões de registros',ylim=(0,max(values)/1e9*1.18))
    ax[1].plot(epochs,np.asarray(df,dtype=float)/1e9,color=COLORS[2],label='full')
    ax[1].set(title='Contador de boost integral',xlabel='Época / intervalo registrado',ylabel='Bilhões de registros por intervalo')
    ax[1].legend()
    save_figure(fig,out,'threshold_sensitivity',plt)

    metrics, targets = [], []
    fig, ax = plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
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
            ax[col].plot(epochs,mean,label=label,color=color)
            ax[col].fill_between(epochs,np.maximum(mean-sd,np.finfo(float).tiny),mean+sd,color=color,alpha=.12)
        ax[col].set(title=group,yscale='log',xlabel='Época registrada',ylabel='Loss registrada · média ± DP entre trajetórias')
        ax[col].legend(fontsize=8)
    fig.suptitle('Curvas de loss dos experimentos\nFaixas = dispersão entre 3 trajetórias',fontsize=14)
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


    loss_rows=[]
    for group,methods in CURVES.items():
        for label,filename in methods:
            rows=[r for r in metrics if r['file']==filename]
            loss_rows.append([group,label,
                f"{np.mean([r['normalized_auc'] for r in rows]):.6f}",
                f"{np.mean([r['last_window_mean'] for r in rows]):.6f}",
                f"{np.mean([r['temporal_sd'] for r in rows]):.6f}"])
    within=float(c4[-1]/c2[-1])
    lines=[
        '# AdaSym: análise dos experimentos e diagnóstico do boost','',
        '**Pesquisa experimental de Ítalo Capuzzo em algoritmos de otimização para machine learning.**','',
        'A pesquisa combina descoberta simbólica, experimentação com redes neurais e análise de dados. '
        'Curvas de loss, varreduras de hiperparâmetros e contadores internos participam da identificação '
        'de padrões e da revisão das hipóteses sobre a regra de atualização.','',
        '## 1. Evolução da fórmula de atualização','',
        'A formulação evolui da expressão descoberta para uma regra com memória ao quadrado, '
        'piso relativo, memória direcional separada e peso temporal. A análise dos resultados '
        'acompanha esse desenvolvimento: investiga tanto o desempenho de treinamento quanto '
        'o mecanismo de amplificação por coordenada.','',
        md_table(['Componente','Questão investigada'],[
            ['Memória direcional','Como a consistência recente dos gradientes participa do passo?'],
            ['Piso v/c','Como limitar a amplificação produzida pela correção?'],
            ['Coeficientes de m2','Como a parametrização da memória afeta o objetivo?'],
            ['Peso temporal','Como a participação da correção evolui durante o treinamento?']]),'',
        '## 2. Boost direcional e threshold','',
        'O AdaSym reduz o denominador da atualização de acordo com a memória de consistência '
        'dos gradientes. Esse mecanismo produz um boost element-wise, limitado pelo piso v/c. '
        'Na configuração examinada, o piso é v/4, com LeNet/Fashion-MNIST durante 150 épocas.','',
        'Os registros c4, c2 e full acompanham o mecanismo de boost. full está associado ao '
        'boost integral limitado pelo threshold; c4 e c2 acompanham faixas de boost parcial '
        'relacionadas a v/4 e v/2. c4 está incluído em c2.','',
        'Os registros são acumulados. Para examinar cada intervalo, utilizam-se diferenças '
        'sucessivas dos contadores. A composição de c2 é dividida em c4 e c2 fora de c4, '
        'evitando dupla contagem.','',
        md_table(['Contador','Quantidade acumulada'],[
            ['c4',f'{totals[0]:,}'],['c2',f'{totals[1]:,}'],
            ['c2 fora de c4',f'{totals[1]-totals[0]:,}'],['full',f'{totals[2]:,}']]),'',
        '![Dinâmica dos contadores](threshold_dynamics.png)','',
        '## 3. Composição e evolução temporal','',
        f'Dentro dos registros de c2, c4 representa **{within:.2%}** e a faixa complementar '
        f'representa **{1-within:.2%}**. Essas proporções descrevem a composição interna de c2.','',
        md_table(['Janela de épocas','c4','c2','full','c4 dentro de c2'],[
            [v['window'],f"{v['c4']:,}",f"{v['c2']:,}",f"{v['full']:,}",f"{v['c4_within_c2']:.3%}"]
            for v in phases]),'',
        'As janelas agregam as contagens antes de calcular a composição. As séries por intervalo '
        'e acumuladas permitem examinar a dinâmica do boost ao longo do treinamento. '
        'A janela móvel reúne até dez intervalos e descreve a evolução das faixas monitoradas.','',
        '![Composição por fase](threshold_phases.png)','',
        '![Contagens do boost](threshold_sensitivity.png)','',
        '## 4. Dinâmica da loss','',
        'As curvas são analisadas por trajetória e por grupo. A AUC normalizada utiliza '
        'a integral trapezoidal entre a primeira e a última posição registrada, dividida pelo '
        'comprimento desse intervalo. O resumo final considera as últimas 20 épocas.','',
        md_table(['Grupo','Método','AUC normalizada','Loss média nas últimas 20','DP temporal médio'],loss_rows),'',
        'A leitura conjunta da curva inteira e do trecho final permite distinguir aceleração '
        'inicial, qualidade tardia e variação temporal. A dispersão entre trajetórias é '
        'apresentada nas faixas do gráfico, enquanto o DP temporal descreve a variação dentro '
        'da janela final de cada trajetória.','',
        '![Diagnóstico das losses](loss_diagnostics.png)','',
        'Os resultados para diferentes alvos e quantidades de épocas consecutivas estão em '
        '`convergence_targets.csv`. O arquivo preserva o resultado de cada trajetória; '
        'o registro de época permanece vazio quando o alvo não é atingido no horizonte observado.','',
        '## 5. Sensibilidade aos hiperparâmetros','',
        'As varreduras permitem examinar a resposta do objetivo à learning rate e aos '
        'coeficientes de memória. As tabelas mantêm cada configuração da grade, possibilitando '
        'análise além do ponto de mínimo.','',
        '![Exploração dos hiperparâmetros](hyperparameter_diagnostics.png)','',
        '## 6. Síntese da análise','',
        'Os dados conectam a evolução da fórmula ao diagnóstico do mecanismo e à comparação '
        'experimental. Os contadores mostram a dinâmica das faixas de boost; as curvas de '
        'loss descrevem o comportamento de treinamento; as varreduras mostram a resposta aos '
        'hiperparâmetros. A combinação dessas perspectivas caracteriza a ciência de dados '
        'como metodologia da pesquisa AdaSym.','',
        '## Reprodução e tabelas','',
        '```bash\npython -m pip install -e ".[plots,docs]"\npython scripts/analyze_research.py\n```','',
        'Os resultados incluem CSVs por época e trajetória, figuras PNG/SVG, manifesto com '
        'hashes dos dados e uma cópia HTML com figuras incorporadas. O notebook '
        '`notebooks/threshold-diagnostics.ipynb` apresenta o diagnóstico executável.'
    ]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    source_files=[Path(__file__),ROOT/'src/optimizer_research/diagnostics.py',ROOT/'data/experiment_schema.json']
    manifest=dict(author='Ítalo Capuzzo',experiment='AdaSym boost diagnostics',
        epochs=len(full),counter_unit='repeated coordinate updates',storage='cumulative',
        subset_relationship='c4 included in c2',normalization='within c2 only',
        targets=[.05,.1,.2],sustained_widths=[1,5,10],late_window=20,
        environment=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
        inputs=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in sorted(data.iterdir()) if p.suffix in ('.npy','.txt')],
        analysis_sources=[dict(file=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for p in source_files],
        counter_totals=dict(c4=totals[0],c2=totals[1],full=totals[2]),
        within_c2_composition=dict(c4=within,outside_c4=1-within))
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    standalone_report(out)
    print(json.dumps(dict(output=str(out),epochs=len(full),counter_totals=totals),ensure_ascii=False))


if __name__ == '__main__':
    main()
