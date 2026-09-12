"""Gera web/brasileirao.html: pagina unica, autocontida, que roda o Monte Carlo
no navegador a partir das distribuicoes de placar exportadas daqui.

O ajuste Dixon-Coles continua sendo feito em Python (scipy); o que vai para a
pagina e a distribuicao conjunta de placares ja pronta de cada jogo restante.
Assim o condicional vira clique, sem servidor e sem reajustar nada.
"""
import json
import numpy as np
import pandas as pd
from model import DixonColes, MAXG
from run import load, fit_model

GMAX = 8  # teto de gols exportado; a cauda acima disso e desprezivel


def main():
    matches, st = load()
    dc, _ = fit_model(matches, None)
    rem = matches[~matches.played].reset_index(drop=True)
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
             "p": [round(float(x), 6) for x in grid[m].ravel()]}
            for m, (h, a) in enumerate(zip(hi, ai))
        ],
        "played": int(matches.played.sum()),
    }

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

    # controle: a grade truncada ainda soma 1 e preserva o 1X2 do modelo cheio?
    full = dc.probs_1x2(hi, ai)
    g = np.arange(GMAX + 1)
    GH, GA = np.meshgrid(g, g, indexing="ij")
    trunc = np.stack([grid[:, GH > GA].sum(1), grid[:, GH == GA].sum(1),
                      grid[:, GH < GA].sum(1)], 1)
    print(f"erro maximo do truncamento em 1X2: {np.abs(full - trunc).max()*100:.4f} p.p.")


if __name__ == "__main__":
    main()
