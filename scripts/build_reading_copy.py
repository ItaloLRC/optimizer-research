"""Build a local, offline HTML reading copy. Requires .[docs]."""
import html
from pathlib import Path
import re
import markdown
from latex2mathml.converter import convert


ROOT = Path(__file__).resolve().parents[1]


def render(path, before=None):
    text = path.read_text(encoding='utf-8')
    if before is not None:
        text = text.split(before,1)[0]
    math_blocks = []
    def replace_math(match):
        math_blocks.append(convert(match.group(1).strip(), display='block'))
        return '\n\nMATHPLACEHOLDER'+str(len(math_blocks)-1)+'END\n\n'
    text = re.sub(r'\$\$(.*?)\$\$', replace_math, text, flags=re.S)
    body = markdown.markdown(text,extensions=['tables','fenced_code'])
    for index,block in enumerate(math_blocks):
        body = body.replace('<p>MATHPLACEHOLDER'+str(index)+'END</p>', '<div class="equation">'+block+'</div>')
    # Links in each markdown file are relative to that file, but this HTML is at root.
    def link(match):
        target = match.group(2)
        if not target.startswith(('http:','https:','#','mailto:')):
            target = (path.parent/target).relative_to(ROOT).as_posix()
        return match.group(1)+'="'+html.escape(target,quote=True)+'"'
    body = re.sub(r'(href|src)="([^"]+)"',link,body)
    return body


def main():
    sections = [('inicio','Visão geral','README.md'),('convergencia','Convergência observada','results/original_tests/convergence.md'),('testes','Testes do autor','results/original_tests/report.md'),('pesquisa','Formulação detalhada','docs/pesquisa.md'),('mapa','Mapa dos testes','docs/mapa_experimentos.md'),('historico','Notas da auditoria','results/historical/report.md'),('novo','Ensaio adicional','results/synthetic/report.md'),('protocolo','Próximos experimentos','docs/protocolo.md'),('verificacao','Verificação','results/verification.md'),('origem','Proveniência','docs/proveniencia.md')]
    navigation = ''.join(f'<a href="#{id}">{label}</a>' for id,label,_ in sections)
    content = []
    for id,label,path in sections:
        content.append(f'<section id="{id}">{render(ROOT/path, before="## Executar" if id=="inicio" else None)}</section>')
        if id == 'historico':
            content.append('<figure><img src="results/historical/historical_curves.png" alt="Curvas históricas recalculadas"><figcaption>Arrays originais: comparação descritiva, split e sementes não identificados.</figcaption></figure><figure><img src="results/historical/beta_grids.png" alt="Grades de betas históricas"><figcaption>Grades preservam a região de valores inválidos.</figcaption></figure>')
        elif id == 'novo':
            content.append('<figure><img src="results/synthetic/summary.png" alt="Experimento sintético e consistência"><figcaption>À esquerda, MLP sintético. À direita, resposta a gradientes controlados.</figcaption></figure>')
    css = '''
:root{color-scheme:light;--ink:#182937;--teal:#006e69;--line:#d5e0e3}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f7f9f9;color:var(--ink);font:17px/1.7 'Segoe UI',Arial,sans-serif}header{background:#142e3b;color:white;padding:55px max(28px,calc((100vw - 1040px)/2)) 42px}header p{max-width:780px;color:#d9e8e8;margin:10px 0}header h1{font-size:46px;line-height:1.15;margin:8px 0 18px}.eyebrow{font-size:13px;letter-spacing:2px;text-transform:uppercase;color:#8dd1c5}nav{padding:20px 28px;background:white;border-bottom:1px solid var(--line);display:flex;justify-content:center;gap:10px 24px;flex-wrap:wrap}nav a{font-size:14px}main{max-width:1100px;margin:auto;padding:30px 40px 70px;background:white}section{padding:32px 0 16px;border-bottom:1px solid var(--line);scroll-margin-top:20px}section h1{font-size:29px;line-height:1.35;margin:0 0 24px}h2{font-size:24px;margin-top:40px;line-height:1.35}h3{font-size:20px;margin-top:30px}a{color:var(--teal);text-underline-offset:3px}p{margin:16px 0}table{border-collapse:collapse;font-size:14px;display:block;overflow-x:auto;width:100%;margin:24px 0}td,th{padding:11px 14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}th{background:#edf4f4}code{font:0.87em Consolas,monospace;background:#edf2f4;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}pre{background:#edf2f4;padding:18px;overflow-x:auto}pre code{padding:0}.equation{padding:18px 12px;margin:20px 0;background:#f3f8f7;overflow-x:auto;font-size:20px}figure{margin:22px 0 40px}img{max-width:100%;height:auto}figcaption{font-size:13px;color:#506875}li{margin:8px 0}footer{color:#536b77;padding:30px;text-align:center;font-size:13px}@media(max-width:700px){header h1{font-size:35px}main{padding:20px}body{font-size:16px}.equation{font-size:16px}}@media print{nav{display:none}header{background:white;color:black;padding:20px}header p{color:black}main{padding:0}.equation{break-inside:avoid}h2,h3{break-after:avoid}}
'''
    document = '<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AdaSym · Janela de consistência</title><style>'+css+'</style></head><body><header><div class="eyebrow">Optimizer Research · AdaSym</div><h1>Uma janela de consistência<br>para adaptar o passo</h1><p>Fundamentação matemática, análise dos testes originais e uma implementação para testar a hipótese.</p><p>Revisão de 29 de outubro de 2026 · Pesquisa exploratória</p></header><nav>'+navigation+'</nav><main>'+''.join(content)+'</main><footer>Entrega local. As fórmulas usam MathML nativo e os gráficos estão nesta pasta. Não requer conexão com a internet.</footer></body></html>'
    (ROOT/'LEIA-ME.html').write_text(document,encoding='utf-8')
    print('LEIA-ME.html generated')


if __name__=='__main__':
    main()
