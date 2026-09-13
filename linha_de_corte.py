"""Quantos pontos salvam? Quantos classificam?

A pagina mostra a distribuicao de pontos de cada TIME. Esta pergunta e outra:
a distribuicao de pontos da POSICAO — quanto faz quem termina em 16o (o
primeiro fora do Z4) e em 6o, seja quem for.

E a "linha de corte", e ela e mais estreita que a de qualquer time, porque nao
depende de quem chega la.

    python linha_de_corte.py --n 100000
"""
import argparse
import numpy as np
import pandas as pd
from model import DixonColes
from run import load, fit_model, simulate

POSICOES = [4, 6, 16, 17]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--sem-erro", action="store_true",
                    help="forcas fixas; o padrao inclui o erro de estimacao")
    args = ap.parse_args()

    matches, st = load()
    dc, _ = fit_model(matches, None)
    teams = dc.teams
    rem = matches[~matches.played].reset_index(drop=True)
    base = {k: st.set_index("team").loc[teams, k].to_numpy()
            for k in ("pts", "v", "gp", "gc")}

    import json
    import os
    usar_bs = (not args.sem_erro) and os.path.exists("data/draws.json")
    modelos = [dc]
    if usar_bs:
        d = json.load(open("data/draws.json", encoding="utf-8"))
        modelos = []
        for dr in d["draws"]:
            m = DixonColes(teams)
            m.attack = pd.Series(dr["att"], index=d["teams"])[teams]
            m.defence = pd.Series(dr["def"], index=d["teams"])[teams]
            m.home_adv, m.rho = dr["home"], d["rho"]
            m.params = np.concatenate([
                m.attack.to_numpy()[:-1], m.defence.to_numpy()[:-1],
                [m.home_adv], [m.rho]])
            modelos.append(m)

    por = max(1, args.n // len(modelos))
    print(f"{len(modelos)} conjunto(s) de forcas x {por:,} temporadas = "
          f"{len(modelos)*por:,}".replace(",", "."))
    rng = np.random.default_rng(2026)
    pontos_pos = {p: [] for p in POSICOES}
    for i, m in enumerate(modelos):
        sim = simulate(m, rem, base, por, seed=int(rng.integers(1 << 30)))
        pts, rank = sim["pts"], sim["rank"]
        ordenado = np.sort(pts, axis=1)[:, ::-1]     # pontos por posicao
        for p in POSICOES:
            pontos_pos[p].append(ordenado[:, p - 1])
        print(f"  {i+1}/{len(modelos)}", end="\r")
    print(" " * 30)

    linhas = []
    for p in POSICOES:
        v = np.concatenate(pontos_pos[p])
        linhas.append({
            "posicao": f"{p}o",
            "media": round(float(v.mean()), 1),
            "mediana": int(np.median(v)),
            "moda": int(np.bincount(v).argmax()),
            "p5": int(np.percentile(v, 5)),
            "p25": int(np.percentile(v, 25)),
            "p75": int(np.percentile(v, 75)),
            "p95": int(np.percentile(v, 95)),
            "desvio": round(float(v.std()), 2),
        })
    resumo = pd.DataFrame(linhas)
    print(resumo.to_string(index=False))

    for p in (16, 6):
        v = np.concatenate(pontos_pos[p])
        cnt = np.bincount(v, minlength=100) / len(v) * 100
        faixa = [g for g in range(len(cnt)) if cnt[g] >= 0.3]
        print(f"\n--- pontos de quem termina em {p}o ---")
        print("pts    prob    acumulada (P de o {}o fazer ATE isso)".format(p))
        acc = 0.0
        for g in faixa:
            acc = cnt[:g + 1].sum()
            barra = "#" * int(round(cnt[g] / max(cnt) * 34))
            print(f"{g:3d}  {cnt[g]:6.2f}%  {acc:6.2f}%  {barra}")
        print(f"\nsalva-se com {int(np.percentile(v, 95))} pontos em 95% das temporadas"
              if p == 16 else
              f"o G6 exige {int(np.percentile(v, 95))} pontos em 95% das temporadas")

    # quantos pontos bastam, com margem
    v16 = np.concatenate(pontos_pos[16])
    v6 = np.concatenate(pontos_pos[6])
    print("\n--- quantos pontos bastam ---")
    print("pts   P(escapar do Z4)   P(entrar no G6)")
    for g in range(38, 75, 2):
        print(f"{g:3d}      {(v16 <= g).mean()*100:6.2f}%           {(v6 <= g).mean()*100:6.2f}%")

    out = pd.DataFrame({"posicao": np.repeat(POSICOES, len(np.concatenate(pontos_pos[4]))),
                        "pontos": np.concatenate([np.concatenate(pontos_pos[p])
                                                  for p in POSICOES])})
    (out.groupby(["posicao", "pontos"]).size()
        .rename("n").reset_index()
        .assign(prob=lambda d: d.n / d.groupby("posicao").n.transform("sum") * 100)
        .to_csv("out/linha_de_corte.csv", index=False, encoding="utf-8"))
    resumo.to_csv("out/linha_de_corte_resumo.csv", index=False, encoding="utf-8")
    print("\ngravado out/linha_de_corte.csv e out/linha_de_corte_resumo.csv")


if __name__ == "__main__":
    main()
