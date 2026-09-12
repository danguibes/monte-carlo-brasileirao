"""O bootstrap muda a conta da meia-vida?

Peso por recencia tem um custo que nao aparece no log-loss: ele JOGA FORA
amostra. Com meia-vida de 50 jogos, um jogo de 100 jogos atras pesa 1/4 de um
jogo de hoje — o tamanho efetivo da amostra encolhe, e as forcas estimadas
ficam mais ruidosas.

Antes do bootstrap nao tinhamos como medir esse custo. Agora temos: basta
refazer o bootstrap parametrico com o peso ligado e ver quanto o erro das
forcas cresce. A pergunta vira uma troca mensuravel:

    o ganho de previsao (+0,0185 nats/jogo, t=0,98) compensa
    o aumento do erro de estimacao?

    python comparar_recencia.py
"""
import numpy as np
import pandas as pd
from model import DixonColes, MAXG
from run import load, fit_model, simulate, G4, Z4
from model import recency_weights

B = 40
MEIAS_VIDAS = [None, 50]


def sample_scores(dc, h, a, rng):
    grid = dc.score_grid(h, a).reshape(len(h), -1)
    cum = grid.cumsum(1) / grid.sum(1, keepdims=True)
    idx = (cum < rng.random(len(h))[:, None]).sum(1)
    return idx // (MAXG + 1), idx % (MAXG + 1)


def n_efetivo(w):
    """Tamanho efetivo da amostra de Kish: (soma w)^2 / soma(w^2)."""
    w = np.asarray(w, float)
    return w.sum() ** 2 / (w ** 2).sum()


def main():
    matches, st = load()
    teams = sorted(matches.home.unique())
    rem = matches[~matches.played].reset_index(drop=True)
    rng = np.random.default_rng(23)
    linhas = []

    for hl in MEIAS_VIDAS:
        dc, (h, a, gh, ga) = fit_model(matches, hl)
        w = recency_weights(len(h), hl)
        nef = n_efetivo(w)

        draws = []
        for _ in range(B):
            sh, sa = sample_scores(dc, h, a, rng)
            draws.append(DixonColes(teams).fit(h, a, sh, sa, w=w))
        att = np.array([d.attack[teams].to_numpy() for d in draws])
        erro = att.std(0).mean()
        amp = float((dc.attack - dc.defence).max() - (dc.attack - dc.defence).min())

        base = {k: st.set_index("team").loc[teams, k].to_numpy()
                for k in ("pts", "v", "gp", "gc")}
        ranks = []
        for d in draws:
            d.rho = dc.rho
            ranks.append(simulate(d, rem, base, 500,
                                  seed=int(rng.integers(1 << 30)))["rank"])
        rk = np.concatenate(ranks, 0)
        nt = len(teams)
        tit = (rk == 1).mean(0) * 100

        linhas.append({
            "peso": "uniforme" if hl is None else f"meia-vida {hl}",
            "jogos": len(h),
            "n_efetivo": round(nef, 1),
            "erro_ataque": round(erro, 4),
            "amplitude": round(amp, 3),
            "ruido_sinal_%": round(erro / amp * 100, 1),
            "titulo_1o_%": round(float(np.sort(tit)[-1]), 1),
            "titulo_2o_%": round(float(np.sort(tit)[-2]), 1),
        })
        print(f"  {linhas[-1]['peso']} pronto", end="\r")

    df = pd.DataFrame(linhas)
    print(" " * 40)
    print(df.to_string(index=False))
    u, r = linhas[0], linhas[1]
    print(f"\nA meia-vida de 50 reduz a amostra efetiva de {u['n_efetivo']:.0f} "
          f"para {r['n_efetivo']:.0f} jogos ({(1-r['n_efetivo']/u['n_efetivo'])*100:.0f}% a menos)")
    print(f"e aumenta o erro das forcas de {u['erro_ataque']:.3f} para "
          f"{r['erro_ataque']:.3f} log ({(r['erro_ataque']/u['erro_ataque']-1)*100:+.0f}%).")
    print(f"ruido/sinal: {u['ruido_sinal_%']}% -> {r['ruido_sinal_%']}%")


if __name__ == "__main__":
    main()
