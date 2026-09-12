"""Backtest TEMPORAL: treina no passado, prevê o futuro — que é o uso real.

Só é possível porque fetch_globo.py trouxe data por jogo. Avança uma janela:
treina nos primeiros K jogos em ordem cronológica, mede log-loss 1X2 nos H
jogos seguintes, repete. Compara meias-vidas diferentes contra peso uniforme e
contra a taxa média da liga.

É o teste que decide se o peso por recência vale alguma coisa. Aviso de
antemão: o esperado é que a diferença seja pequena e possivelmente nula — e
saber disso é o ponto.
"""
import numpy as np
import pandas as pd
from model import DixonColes

MIN_TREINO, HORIZONTE = 150, 20
MEIAS_VIDAS = [None, 120, 80, 50, 30]


def logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


def main():
    m = pd.read_csv("data/matches.csv", encoding="utf-8")
    if "data" not in m.columns:
        raise SystemExit("sem coluna de data: rode fetch_globo.py primeiro")
    p = m[m.played].sort_values("data").reset_index(drop=True)
    teams = sorted(m.home.unique())
    base = DixonColes(teams)
    H = p.home.map(base.idx).to_numpy()
    A = p.away.map(base.idx).to_numpy()
    GH = p.home_goals.to_numpy(int)
    GA = p.away_goals.to_numpy(int)
    Y = np.where(GH > GA, 0, np.where(GH == GA, 1, 2))

    cortes = list(range(MIN_TREINO, len(p) - HORIZONTE + 1, HORIZONTE))
    print(f"{len(p)} jogos com data. {len(cortes)} janelas, "
          f"treino >= {MIN_TREINO}, horizonte {HORIZONTE} jogos.\n")

    perdas = {str(hl): [] for hl in MEIAS_VIDAS}
    perdas["taxa_liga"] = []
    for k in cortes:
        tr = slice(0, k)
        te = slice(k, k + HORIZONTE)
        for hl in MEIAS_VIDAS:
            if hl is None:
                w = None
            else:
                idade = k - 1 - np.arange(k)
                w = 0.5 ** (idade / hl)
            dc = DixonColes(teams).fit(H[tr], A[tr], GH[tr], GA[tr], w=w)
            perdas[str(hl)] += list(logloss(dc.probs_1x2(H[te], A[te]), Y[te]))
        taxa = np.bincount(Y[tr], minlength=3) / k
        perdas["taxa_liga"] += list(logloss(np.tile(taxa, (HORIZONTE, 1)), Y[te]))
        print(f"  janela ate o jogo {k}", end="\r")

    df = pd.DataFrame(perdas)
    res = pd.DataFrame({
        "log_loss": df.mean(),
        "erro_padrao": df.std(ddof=1) / np.sqrt(len(df)),
    })
    res.index = ["uniforme" if i == "None" else
                 ("taxa media da liga" if i == "taxa_liga" else f"meia-vida {i} jogos")
                 for i in res.index]
    print(" " * 40)
    print(f"log-loss 1X2 fora da amostra, no futuro ({len(df)} previsoes)\n")
    print(res.round(4).to_string())

    d = df["taxa_liga"] - df["None"]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"\nDixon-Coles uniforme vs taxa da liga: {d.mean():+.4f} nats/jogo (t={t:.2f})")
    melhor = min((hl for hl in MEIAS_VIDAS if hl is not None),
                 key=lambda hl: df[str(hl)].mean())
    d2 = df["None"] - df[str(melhor)]
    t2 = d2.mean() / (d2.std(ddof=1) / np.sqrt(len(d2)))
    print(f"melhor meia-vida ({melhor}) vs uniforme:  {d2.mean():+.4f} nats/jogo (t={t2:.2f})")
    print("\nt abaixo de ~2 quer dizer que a diferenca nao se distingue de ruido.")


if __name__ == "__main__":
    main()
