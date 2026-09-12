"""Gera web/brasileirao.html: pagina unica, autocontida, que roda o Monte Carlo
no navegador a partir das distribuicoes de placar exportadas daqui.

O ajuste Dixon-Coles continua sendo feito em Python (scipy); o que vai para a
pagina e a distribuicao conjunta de placares ja pronta de cada jogo restante.
Assim o condicional vira clique, sem servidor e sem reajustar nada.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from model import DixonColes, MAXG
from run import load, fit_model

GMAX = 8  # teto de gols exportado; a cauda acima disso e desprezivel


def main():
    matches, st = load()
    dc, _ = fit_model(matches, None)
    rem = matches[~matches.played].copy()
    if "data" in rem.columns:
        # ordem cronologica; adiados (sem data) caem no fim
        rem = (rem.assign(_o=rem.data.fillna("9999"))
                  .sort_values(["_o", "home"]).drop(columns="_o"))
    rem = rem.reset_index(drop=True)
    hi = rem.home.map(dc.idx).to_numpy()
    ai = rem.away.map(dc.idx).to_numpy()

    grid = dc.score_grid(hi, ai)[:, :GMAX + 1, :GMAX + 1]
    grid = grid / grid.sum(axis=(1, 2), keepdims=True)

    base = st.set_index("team").loc[dc.teams]
    payload = {
        "teams": dc.teams,
        "gmax": GMAX,
        "homeAdv": round(float(np.exp(dc.home_adv)), 3),
        "rho": round(float(dc.rho), 3),
        "base": {
            "pts": base.pts.astype(int).tolist(),
            "v": base.v.astype(int).tolist(),
            "gp": base.gp.astype(int).tolist(),
            "gc": base.gc.astype(int).tolist(),
            "j": base.j.astype(int).tolist(),
        },
        "strength": {
            "attack": [round(float(x), 3) for x in dc.attack[dc.teams]],
            "defence": [round(float(x), 3) for x in dc.defence[dc.teams]],
        },
        "fixtures": [
            {"h": int(h), "a": int(a),
             "d": (rem.data.iloc[m] if pd.notna(rem.data.iloc[m]) else None)
                  if "data" in rem.columns else None,
             "r": int(rem.rodada.iloc[m]) if "rodada" in rem.columns else None,
             "p": [round(float(x), 6) for x in grid[m].ravel()]}
            for m, (h, a) in enumerate(zip(hi, ai))
        ],
        "played": int(matches.played.sum()),
    }

    # Sorteios de parametro do bootstrap, se existirem. Sao eles que permitem a
    # pagina embutir o erro de estimacao das forcas em vez de tratar ataque e
    # defesa como se fossem conhecidos exatamente.
    if os.path.exists("data/draws.json"):
        d = json.load(open("data/draws.json", encoding="utf-8"))
        if d["teams"] != dc.teams:
            sys.exit("data/draws.json e de outro conjunto de times; "
                     "rode bootstrap.py de novo")
        payload["draws"] = d["draws"]
        att = np.array([x["att"] for x in d["draws"]])
        payload["erro_ataque"] = float(att.std(0).mean())
        print(f"  + {len(d['draws'])} sorteios de parametro (erro de estimacao)")
    else:
        print("  (sem data/draws.json: a pagina sai sem erro de estimacao; "
              "rode bootstrap.py)")

    tpl = open("web/template.html", encoding="utf-8").read()
    html = tpl.replace("/*__DADOS__*/null",
                       json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    # dois nomes, mesmo conteudo: index.html e o que o GitHub Pages serve na
    # raiz; brasileirao.html e o que faz sentido receber por WhatsApp.
    open("web/brasileirao.html", "w", encoding="utf-8").write(html)
    open("web/index.html", "w", encoding="utf-8").write(html)
    kb = len(html) / 1024
    print(f"web/index.html + brasileirao.html  {kb:.0f} KB  "
          f"({len(payload['fixtures'])} jogos restantes, grade {GMAX+1}x{GMAX+1})")

    glossario(dc, matches, rem, grid, payload)

    # controle: a grade truncada ainda soma 1 e preserva o 1X2 do modelo cheio?
    full = dc.probs_1x2(hi, ai)
    g = np.arange(GMAX + 1)
    GH, GA = np.meshgrid(g, g, indexing="ij")
    trunc = np.stack([grid[:, GH > GA].sum(1), grid[:, GH == GA].sum(1),
                      grid[:, GH < GA].sum(1)], 1)
    print(f"erro maximo do truncamento em 1X2: {np.abs(full - trunc).max()*100:.4f} p.p.")


def glossario(dc, matches, rem, grid, payload):
    """O glossario e GERADO, nao escrito a mao: os numeros dele sao os da
    rodada. Texto fixo com numero velho e pior que numero nenhum."""
    p = matches[matches.played]
    gh = p.home_goals.to_numpy(int)
    ga = p.away_goals.to_numpy(int)
    hi = rem.home.map(dc.idx).to_numpy()
    ai = rem.away.map(dc.idx).to_numpy()
    p1x2 = dc.probs_1x2(hi, ai)

    subs = {
        "__JOGOS__": f"{len(p)}",
        "__CASA__": f"{np.exp(dc.home_adv):.3f}".replace(".", ","),
        "__RHO__": f"{dc.rho:+.3f}".replace(".", ","),
        "__RHOPCT__": f"{-dc.rho*100:.1f}".replace(".", ","),
        "__PCTM__": f"{(gh > ga).mean()*100:.0f}",
        "__PCTE__": f"{(gh == ga).mean()*100:.0f}",
        "__PCTV__": f"{(gh < ga).mean()*100:.0f}",
        "__NDRAWS__": f"{len(payload.get('draws', []))}",
        "__ERROPCT__": f"{(np.exp(payload.get('erro_ataque', 0)) - 1)*100:.0f}",
        "__PEMP__": f"{p1x2[:,1].mean()*100:.1f}".replace(".", ","),
        "__PMAND__": f"{p1x2[:,0].mean()*100:.1f}".replace(".", ","),
        "__NEMP__": f"{int((p1x2.argmax(1) == 1).sum())}",
        "__NREM__": f"{len(hi)}",
    }
    html = open("web/glossario_template.html", encoding="utf-8").read()
    for k, v in subs.items():
        html = html.replace(k, v)
    faltou = [k for k in subs if k in html]
    if faltou:
        sys.exit(f"placeholder nao substituido no glossario: {faltou}")
    open("web/glossario.html", "w", encoding="utf-8").write(html)
    print(f"  + web/glossario.html ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
