"""Controle: o Dixon-Coles bate baselines fora da amostra?

Divide os jogos ja disputados em treino/teste ao acaso, ajusta so no treino e
mede log-loss em 1X2 no teste, contra tres baselines:
  base_liga  - taxa media M/E/V da liga (sem saber quem joga)
  base_casa  - sempre 100% mandante, truncado (mostra o piso)
  uniforme   - 1/3, 1/3, 1/3

A fonte nao traz datas, entao o corte e aleatorio e mede generalizacao para
outros confrontos, nao extrapolacao no tempo. Vale como controle, nao como
prova de que o modelo preve o futuro.
"""
import numpy as np
import pandas as pd
from model import DixonColes

REP, TEST_FRAC = 20, 0.25


def logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()


def main():
    m = pd.read_csv("data/matches.csv", encoding="utf-8")
    p = m[m.played].reset_index(drop=True)
    teams = sorted(m.home.unique())
    dc0 = DixonColes(teams)
    H = p.home.map(dc0.idx).to_numpy()
    A = p.away.map(dc0.idx).to_numpy()
    GH = p.home_goals.to_numpy(int)
    GA = p.away_goals.to_numpy(int)
    Y = np.where(GH > GA, 0, np.where(GH == GA, 1, 2))

    rows = []
    rng = np.random.default_rng(7)
    for r in range(REP):
        idx = rng.permutation(len(p))
        ntest = int(len(p) * TEST_FRAC)
        te, tr = idx[:ntest], idx[ntest:]
        dc = DixonColes(teams).fit(H[tr], A[tr], GH[tr], GA[tr])
        pm = dc.probs_1x2(H[te], A[te])
        rate = np.bincount(Y[tr], minlength=3) / len(tr)
        rows.append({
            "dixon_coles": logloss(pm, Y[te]),
            "base_liga": logloss(np.tile(rate, (ntest, 1)), Y[te]),
            "uniforme": logloss(np.full((ntest, 3), 1 / 3), Y[te]),
        })
        print(f"  rep {r+1}/{REP}", end="\r")

    df = pd.DataFrame(rows)
    print("\nlog-loss fora da amostra (menor e melhor), "
          f"{REP} reparticoes, {int(len(p)*TEST_FRAC)} jogos de teste cada\n")
    print(df.agg(["mean", "std"]).T.round(4).to_string())
    d = df.base_liga - df.dixon_coles
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"\nganho sobre base_liga: {d.mean():+.4f} nats/jogo "
          f"(t={t:.2f}, vence em {(d>0).sum()}/{len(d)} reparticoes)")


if __name__ == "__main__":
    main()
