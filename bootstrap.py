"""A falha que o modelo tem e que nao aparece em nenhum grafico ate agora.

O Monte Carlo sorteia o PLACAR de cada jogo, mas trata a forca de cada time
como se fosse conhecida exatamente. Nao e: ataque e defesa sao estimados a
partir de ~26 jogos, e carregam erro. Ignorar esse erro deixa a projecao mais
confiante do que ela tem direito de ser — o campeao parece mais certo do que e.

Bootstrap parametrico: simula uma temporada inteira a partir do modelo
ajustado, reajusta o modelo NELA, e guarda os parametros. Repetindo, temos a
distribuicao amostral das forcas. Depois compara as probabilidades finais com
e sem esse erro embutido.

    python bootstrap.py --b 60 --n 30000
"""
import argparse
import json
import numpy as np
import pandas as pd
from model import DixonColes, MAXG
from run import load, fit_model, simulate, report, G4, Z4


def sample_scores(dc, h, a, rng):
    grid = dc.score_grid(h, a).reshape(len(h), -1)
    cum = grid.cumsum(1) / grid.sum(1, keepdims=True)
    u = rng.random(len(h))
    idx = (cum < u[:, None]).sum(1)
    return idx // (MAXG + 1), idx % (MAXG + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=60, help="reajustes do bootstrap")
    ap.add_argument("--n", type=int, default=30000, help="temporadas no total")
    args = ap.parse_args()

    matches, st = load()
    dc, (h, a, gh, ga) = fit_model(matches, None)
    teams = dc.teams
    rem = matches[~matches.played].reset_index(drop=True)
    base = {k: st.set_index("team").loc[teams, k].to_numpy()
            for k in ("pts", "v", "gp", "gc")}
    rng = np.random.default_rng(11)

    print(f"bootstrap parametrico: {args.b} reajustes sobre {len(h)} jogos")
    draws = []
    for b in range(args.b):
        sh, sa = sample_scores(dc, h, a, rng)
        d = DixonColes(teams).fit(h, a, sh, sa)
        draws.append(d)
        print(f"  {b+1}/{args.b}", end="\r")
    print(" " * 30)

    att = np.array([d.attack[teams].to_numpy() for d in draws])
    dfn = np.array([d.defence[teams].to_numpy() for d in draws])
    ep = pd.DataFrame({
        "time": teams,
        "ataque": dc.attack[teams].to_numpy().round(3),
        "erro_ataque": att.std(0).round(3),
        "defesa": dc.defence[teams].to_numpy().round(3),
        "erro_defesa": dfn.std(0).round(3),
    })
    print("erro-padrao das forcas estimadas (log):")
    print(ep.to_string(index=False))
    print(f"\nerro tipico do ataque: {att.std(0).mean():.3f} log "
          f"= {(np.exp(att.std(0).mean())-1)*100:.0f}% em gols")
    amp = (dc.attack - dc.defence).max() - (dc.attack - dc.defence).min()
    print(f"amplitude de forca entre os times: {amp:.3f} log")
    print(f"razao ruido/sinal: {att.std(0).mean()/amp*100:.0f}%")

    # --- probabilidades COM e SEM o erro de estimacao ---
    fixo = simulate(dc, rem, base, args.n, seed=5)
    por = max(1, args.n // args.b)
    ranks = []
    for d in draws:
        d.rho = dc.rho
        ranks.append(simulate(d, rem, base, por, seed=int(rng.integers(1 << 30)))["rank"])
    rank_bs = np.concatenate(ranks, 0)

    nt = len(teams)
    out = pd.DataFrame({
        "time": teams,
        "titulo_fixo": (fixo["rank"] == 1).mean(0) * 100,
        "titulo_bs": (rank_bs == 1).mean(0) * 100,
        "G4_fixo": (fixo["rank"] <= G4).mean(0) * 100,
        "G4_bs": (rank_bs <= G4).mean(0) * 100,
        "Z4_fixo": (fixo["rank"] >= nt - Z4 + 1).mean(0) * 100,
        "Z4_bs": (rank_bs >= nt - Z4 + 1).mean(0) * 100,
    }).round(2)
    out["d_titulo"] = (out.titulo_bs - out.titulo_fixo).round(2)
    out = out.sort_values("titulo_fixo", ascending=False)
    print("\nprobabilidades sem e com o erro de estimacao das forcas:")
    print(out[out.titulo_fixo + out.Z4_fixo > 0.5].to_string(index=False))

    ep.to_csv("out/erro_forcas.csv", index=False, encoding="utf-8")
    out.to_csv("out/bootstrap.csv", index=False, encoding="utf-8")
    json.dump({"att_sd": att.std(0).tolist(), "def_sd": dfn.std(0).tolist(),
               "teams": teams}, open("data/erro_forcas.json", "w"), indent=1)
    # Os sorteios em si: e com eles que a pagina embute o erro de estimacao.
    # So as forcas e a vantagem de casa; rho fica o do ajuste principal.
    json.dump({
        "teams": teams,
        "rho": round(float(dc.rho), 5),
        "draws": [{"att": [round(float(x), 5) for x in d.attack[teams]],
                   "def": [round(float(x), 5) for x in d.defence[teams]],
                   "home": round(float(d.home_adv), 5)} for d in draws],
    }, open("data/draws.json", "w"), separators=(",", ":"))
    print(f"gravado data/draws.json com {len(draws)} sorteios de parametro")
    print("\ngravado out/bootstrap.csv e out/erro_forcas.csv")


if __name__ == "__main__":
    main()
