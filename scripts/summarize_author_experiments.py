"""Present the author's experiments with conditions transcribed from metadata.

This script only analyzes supplied results. It does not run new training.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/original'
OUT = ROOT/'results/original_tests'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    arrays={p.name:np.load(p,allow_pickle=False) for p in DATA.glob('*.npy')}
    stats={}
    for name,a in arrays.items():
        if a.shape in ((3,100),(3,150)):
            stats[name]=dict(shape=list(a.shape),row_mean=a.mean(1).tolist(),row_last=a[:,-1].tolist(),row_last10=a[:,-10:].mean(1).tolist(),mean=float(a.mean()),last10_mean=float(a[:,-10:].mean()))
    lrgrid=np.concatenate([np.arange(1e-4,1e-3,1e-4),np.arange(1e-3,1e-2,1e-3),np.arange(1e-2,1e-1,1e-2)])
    scans={}
    for name in ('lrs_aucloss.npy','lrs_aucloss_adamW.npy','lrs_AdaSym_aucloss.npy','lim_aucloss.npy'):
        a=arrays[name]; index=int(np.argmin(a))
        scans[name]=dict(size=len(a),best_index=index,minimum=float(a[index]),lr_from_metadata_grid=float(lrgrid[index]) if len(a)==27 else None)
    full=arrays['AdaSym_fminist_full.npy']
    counts={key:dict(final=int(arrays[f'AdaSym_fminist_{key}.npy'][-1]),fraction=float(arrays[f'AdaSym_fminist_{key}.npy'][-1]/full[-1])) for key in ('c2','c4')}
    payload=dict(attribution='Experiments performed by the author; supplied arrays analyzed in this revision',curves=stats,scans=scans,counts=counts,lr_axis_status='27-point axis inferred from metadata; 80-point axis unavailable',fashion100_row_mapping='AdaSym, AdaSymSigmoid, AdamW inferred from slides 8 and 9')
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    lines=['# Testes realizados pelo autor', '', 'Esta seção incorpora os experimentos já realizados na pesquisa AdaSym. As condições vêm de `data/original/metadata.txt` e da apresentação AdaSym. Os números abaixo foram recalculados a partir dos arrays fornecidos. A revisão não executou novamente esses treinamentos.', '', 'A loss é apresentada conforme registrada. Nas matrizes 3 × 150, as tabelas agregam as três trajetórias, como os gráficos de média e desvio padrão dos slides. No arquivo 3 × 100, cada linha corresponde a um otimizador. A média da loss ao longo das épocas descreve a curva inteira e não equivale à loss final, à acurácia nem à AUC original dos vetores de varredura.', '', '## 1. Fashion-MNIST: AdaSym, AdaSymSigmoid e AdamW', '', '**Condição registrada:** Fashion-MNIST, 100 épocas. Arquivo `losses_FashionMNIST.npy`, slides 8 e 9. A ordem das linhas abaixo é inferida pela correspondência dos valores com esses gráficos.', '', '| Método | Loss na época 100 | Média das últimas 10 épocas | Média nas 100 épocas |', '|---|---:|---:|---:|']
    a=arrays['losses_FashionMNIST.npy']
    for i,label in enumerate(('AdaSym','AdaSymSigmoid','AdamW')):
        lines.append(f'| {label} | {a[i,-1]:.6f} | {a[i,-10:].mean():.6f} | {a[i].mean():.6f} |')
    lines += ['', '**Leitura:** AdaSymSigmoid apresenta a menor média da loss ao longo das épocas e nas últimas dez épocas desse teste. A diferença entre AdaSym sem sigmoide e AdamW depende do resumo usado: AdaSym tem média da curva menor, mas AdamW tem média ligeiramente menor nas últimas dez épocas. Isso favorece uma comparação por trajetória, além de um único ponto final.', '', '## 2. LeNet em Fashion-MNIST: comparação de 150 épocas', '', '**Condições no metadata:** LeNet, Fashion-MNIST, 150 épocas e LR 1e-3. O texto cita uma execução sem redução por sigmoide ou exponencial e, em outro trecho, piso v/4, valores (0,75; 0,35) e sigmoide por época. A associação exata desses trechos às variantes deve seguir o código original quando ele estiver disponível. O arquivo `AdaSym_loss_fmnist_nored.npy` identifica explicitamente a variante sem redução temporal pelo nome.', '', '| Resultado salvo | Média das últimas 10 épocas | Média nas 150 épocas |', '|---|---:|---:|']
    for name,label in [('AdamW_losses.npy','AdamW'),('AdaSym_losses.npy','AdaSym'),('AdaSym_loss_fmnist_nored.npy','AdaSym, arquivo nored')]:
        s=stats[name];lines.append(f"| {label} | {s['last10_mean']:.6f} | {s['mean']:.6f} |")
    lines += ['', '**Leitura:** as duas curvas AdaSym registram médias menores que AdamW nesse conjunto de arquivos, tanto na curva inteira quanto no trecho final. O teste documenta a comparação do autor; a interpretação como ganho de generalização depende de identificar se a loss é de treino, validação ou teste.', '', '## 3. CIFAR-10: AdaSymSigmoid contra AdamW', '', '**Condições no metadata:** CIFAR-10, ResNet-20, 150 épocas, LR 1e-3, weight decay 1e-2, sigmoide com transição na época 15 e piso v/8. Arquivos `AdamW_losses_cifar10.npy` e `AdaSymSig_losses_cifar10.npy`, slides 12 e 13.', '', '| Método | Média das últimas 10 épocas | Média nas 150 épocas |', '|---|---:|---:|']
    for name,label in [('AdamW_losses_cifar10.npy','AdamW'),('AdaSymSig_losses_cifar10.npy','AdaSymSigmoid')]:
        s=stats[name];lines.append(f"| {label} | {s['last10_mean']:.6f} | {s['mean']:.6f} |")
    lines += ['', '**Leitura:** AdaSymSigmoid tem menor média de loss nos dois resumos. As curvas preservam a melhoria inicial e a aproximação entre métodos no fim do treinamento, conforme os slides. Não há uma curva de acurácia correspondente nesses arquivos.', '', '## 4. Controle de tamanho do passo: AdamW com LR 2×', '', '**Condição relatada:** LeNet, 150 épocas e AdamW com o dobro da LR. O metadata diz MNIST e os slides 14 e 15 dizem Fashion-MNIST. Mantém-se essa diferença de rótulo sem alterar os dados. `Adam_losses2x.npy` é o nome original do arquivo.', '', '| Método / arquivo | Média das últimas 10 épocas | Média nas 150 épocas |', '|---|---:|---:|']
    for name,label in [('Adam_losses2x.npy','AdamW LR 2×, rótulo dos slides'),('AdaSym_losses.npy','AdaSym')]:
        s=stats[name];lines.append(f"| {label} | {s['last10_mean']:.6f} | {s['mean']:.6f} |")
    lines += ['', '**Leitura:** o controle LR 2× tem menor média sobre toda a curva, enquanto AdaSym tem menor média no trecho final. O próprio histórico do autor já permite perguntar se a vantagem desejada é aceleração inicial, qualidade tardia ou ambas. Esse controle deve acompanhar qualquer conclusão sobre a janela de consistência.', '', '![Curvas dos testes realizados pelo autor](author_training.png)', '', '## 5. Varreduras de learning rate', '', '**Varredura AdaSym de 80 posições:** o metadata informa 300 épocas, 10% de Fashion-MNIST, AUC de loss, WD 1e-2 e valores b1/b2 = (0,75; 0,35). O papel desses dois coeficientes é preservado conforme o texto, sem atribuí-los automaticamente aos momentos do Adam. Arquivo `lrs_AdaSym_aucloss.npy`.', '', f"O menor valor é **{scans['lrs_AdaSym_aucloss.npy']['minimum']:.6f}**, no índice **{scans['lrs_AdaSym_aucloss.npy']['best_index']}** (índice zero). Como a grade de 80 LRs não acompanha o material, esse mínimo não recebe uma LR inventada.", '', '**Varredura comparativa de 27 posições:** `lrs_aucloss.npy` e `lrs_aucloss_adamW.npy`. O metadata fornece uma grade de 27 LRs de 1e-4 a 9e-2. O trecho próximo inclui `sigmoid(15 − época)` e piso v/8, mas não especifica todas as condições dessa rodada.', '', '| Vetor | Mínimo registrado | Índice zero | LR pela grade do metadata |', '|---|---:|---:|---:|']
    for name in ('lrs_aucloss.npy','lrs_aucloss_adamW.npy'):
        s=scans[name];lines.append(f"| {name} | {s['minimum']:.6f} | {s['best_index']} | {s['lr_from_metadata_grid']:g} |")
    lines += ['', 'A correspondência dos 27 pontos com essa grade é uma inferência apoiada pelo tamanho do vetor e pela descrição. O primeiro vetor é associado à variante AdaSym pelo contexto do metadata. O slide 17 usa o rótulo “média das últimas 15 losses”, enquanto o texto descreve AUC e uma faixa maior de LRs. Por isso, o gráfico novo abaixo mostra os valores salvos com seu significado declarado no metadata, sem afirmar que reproduz exatamente o pós-processamento daquele slide.', '', '## 6. Momento de transição da sigmoide', '', '**Condições:** LeNet, 20% de Fashion-MNIST, 150 épocas, LR 1e-3 e WD 1e-2. O metadata descreve a varredura do ponto médio da sigmoide na faixa 0–50. Arquivo `lim_aucloss.npy`, com 50 valores.', '', f"O menor valor registrado é **{scans['lim_aucloss.npy']['minimum']:.6f}**, no índice **{scans['lim_aucloss.npy']['best_index']}**. O eixo exato com seus endpoints não foi salvo. O resultado é identificado pelo índice; se a grade original era 0, 1, …, 49, esse ponto corresponde à transição na época 17.", '', '**Leitura:** esse teste investiga por quanto tempo manter o incentivo à consistência. A sigmoide começa próxima de 1, passa por 0,5 no ponto de transição e se aproxima de zero depois dele.', '', '![Varreduras de LR e transição da sigmoide](author_sweeps.png)', '', '## 7. Busca dos coeficientes de m2', '', 'O metadata identifica duas grades para procurar os melhores valores de beta3 e beta4 de m2. Os mínimos recalculados coincidem com as marcações dos slides 18 e 19.', '', '| Grade | Shape | Mínimo finito | beta3 | beta4 | Valores inválidos |', '|---|---|---:|---:|---:|---:|']
    for name,e3,e4 in [('loss_matrix.npy',(-.3,0),(-3,0)),('loss_matrix (1).npy',(-.3,-.01),(-1.5,0))]:
        a=arrays[name];idx=np.unravel_index(np.nanargmin(a),a.shape)
        lines.append(f'| {name} | 10 × 70 | {a[idx]:.6f} | {10**np.linspace(*e3,10)[idx[0]]:.6f} | {10**np.linspace(*e4,70)[idx[1]]:.6f} | {int((~np.isfinite(a)).sum())} |')
    lines += ['', 'As faixas exatas estão no mapa dos experimentos. Esses mínimos descrevem a varredura do autor. Sem a recorrência de m2 e a definição completa do fitness, os coeficientes não são convertidos em uma nova EMA nem recomendados como configuração universal.', '', '## 8. Frequência de ativação do piso', '', '**Condições:** LeNet, Fashion-MNIST, piso v/4. Arquivos `AdaSym_fminist_c2.npy`, `AdaSym_fminist_c4.npy` e `AdaSym_fminist_full.npy`. O metadata cita 1550 épocas, mas os vetores têm 150 posições, como o eixo do slide 16.', '', '| Contador | Interpretação pelo slide e metadata | Razão acumulada final |', '|---|---|---:|',f"| c4 / full | Condição associada ao piso v/4 | {100*counts['c4']['fraction']:.4f}% |",f"| c2 / full | Condição associada a ≤v/2 | {100*counts['c2']['fraction']:.4f}% |", '', 'O gráfico do slide 16 identifica as séries como acumuladas. A frequência em uma época isolada deve usar o incremento do contador dividido pelo incremento de full. A razão acumulada final mostra quanto o mecanismo de piso participou do treinamento, e não uma fração de runs ou de épocas.', '', '![Frequência acumulada e por posição do registro](author_floor.png)', '', '## 9. Varredura do divisor do piso', '', 'O metadata registra um teste `div_aucloss`: LeNet, 20% de Fashion-MNIST, 150 épocas, LR 1e-3, WD 1e-2, sem decaimento temporal e divisores de 2 a 20. O teste faz parte do percurso relatado pelo autor. O arquivo correspondente não foi fornecido, portanto seu resultado numérico permanece pendente de recuperação.', '', '## Síntese dos testes do autor', '', 'Os dados documentam comparações entre AdaSym, AdaSymSigmoid e AdamW, um controle de LR 2×, varreduras de LR, do ponto de transição da sigmoide e dos coeficientes de m2, além da frequência de acionamento do piso. A variante sigmoide apresenta losses menores nos resumos dos testes de 100 épocas em Fashion-MNIST e de 150 épocas em CIFAR-10. O controle LR 2× explicita um compromisso entre a curva inteira e seu trecho final.', '', 'Esses são resultados existentes da pesquisa. A identificação de splits, seeds e recorrências é necessária para aprofundar as conclusões, mas não elimina os experimentos realizados. A auditoria técnica e as diferenças de metadata permanecem em `results/historical/report.md`. O ensaio sintético desta revisão está documentado separadamente e não substitui os testes do autor.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(arrays,lrgrid)
    print('Author results generated from 16 supplied arrays.')


def plot(arrays,lrgrid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),constrained_layout=True)
    for i,name in enumerate(('AdaSym','AdaSymSigmoid','AdamW')):
        axes[0].plot(np.arange(1,101),arrays['losses_FashionMNIST.npy'][i],label=name)
    for filename,label in [('AdamW_losses.npy','AdamW'),('AdaSym_losses.npy','AdaSym'),('Adam_losses2x.npy','AdamW LR 2× (slides)')]:
        axes[1].plot(np.arange(1,151),arrays[filename].mean(0),label=label)
    for filename,label in [('AdamW_losses_cifar10.npy','AdamW'),('AdaSymSig_losses_cifar10.npy','AdaSymSigmoid')]:
        axes[2].plot(np.arange(1,151),arrays[filename].mean(0),label=label)
    for axis,title in zip(axes,['Fashion-MNIST · 100 épocas','LeNet · 150 épocas','CIFAR-10 / ResNet-20 · 150 épocas']):
        axis.set(xlabel='Época',ylabel='Loss registrada',title=title)
        axis.legend(fontsize=8)
    fig.suptitle('Testes do autor · dados fornecidos; condições descritas no metadata',fontsize=12)
    fig.savefig(OUT/'author_training.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),constrained_layout=True)
    for name,label in [('lrs_aucloss.npy','AdaSym (contexto do metadata)'),('lrs_aucloss_adamW.npy','AdamW')]:
        axes[0].plot(lrgrid,arrays[name],marker='.',label=label)
    axes[0].set(xscale='log',yscale='log',xlabel='LR · eixo reconstruído do metadata',ylabel='Valor salvo (AUC, metadata)',title='Varredura de LR · 27 pontos')
    axes[0].legend(fontsize=8)
    axes[1].plot(np.arange(80),arrays['lrs_AdaSym_aucloss.npy'])
    axes[1].set(yscale='log',xlabel='Índice na grade · LR não disponível',ylabel='AUC da loss registrada',title='AdaSym · 300 épocas / 10% Fashion-MNIST')
    a=arrays['lim_aucloss.npy'];idx=int(a.argmin())
    axes[2].plot(np.arange(len(a)),a)
    axes[2].scatter([idx],[a[idx]],label=f'Mínimo: índice {idx}',zorder=3)
    axes[2].set(xlabel='Índice do ponto de transição',ylabel='AUC da loss registrada',title='Sigmoide · LeNet / 20% Fashion-MNIST')
    axes[2].legend(fontsize=8)
    fig.suptitle('Varreduras do autor · valores e eixos conforme a disponibilidade dos arquivos',fontsize=12)
    fig.savefig(OUT/'author_sweeps.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    full=arrays['AdaSym_fminist_full.npy']
    for key,label in [('c4','Condição ≤ v/4'),('c2','Condição ≤ v/2')]:
        a=arrays[f'AdaSym_fminist_{key}.npy']
        axes[0].plot(np.arange(1,151),100*a/full,label=label)
        axes[1].plot(np.arange(1,151),100*np.diff(np.r_[0,a])/np.diff(np.r_[0,full]),label=label)
    for axis,title in zip(axes,['Razão acumulada · como no slide 16','Razão por incremento do registro']):
        axis.set(xlabel='Posição no registro / época dos slides',ylabel='Percentual (%)',title=title)
        axis.legend(fontsize=8)
    fig.suptitle('Contadores do autor · LeNet / Fashion-MNIST / piso v/4',fontsize=12)
    fig.savefig(OUT/'author_floor.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    main()
