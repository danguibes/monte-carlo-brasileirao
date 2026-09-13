"""O mercado como REGUA, nao como ingrediente.

Odds de fechamento sao o melhor preditor publico que existe para futebol. Elas
nao entram no modelo: servem para medir quanto do erro e irredutivel e quanto e
meu. Sem isso, saber que o modelo empata com um baseline burro nao diz se o teto
esta perto ou longe.

Fonte: football-data.co.uk /new/BRA.csv — sem chave, temporadas 2021+, com
resultado E odds de fechamento no mesmo arquivo. Como ela traz os placares,
o modelo pode ser reajustado nela mesma, temporada a temporada, sem depender
do mapa de nomes.

Protocolo: caminhada para a frente DENTRO de cada temporada. Para cada bloco de
jogos, o modelo ve apenas o que veio antes — mesma condicao do mercado, que
tambem so conhece o passado no momento em que fecha a odd.

    python regua_mercado.py
"""
import io
import sys

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq

from model import DixonColes

URL = "https://www.football-data.co.uk/new/BRA.csv"
UA = {"User-Agent": "Mozilla/5.0 (brasileirao-mc; uso pessoal)"}
MIN_TREINO, BLOCO = 140, 20
CACHE = "data/odds_bra.csv"


def baixar(refresh=False):
    import os
    if os.path.exists(CACHE) and not refresh:
        return pd.read_csv(CACHE, encoding="utf-8")
    r = requests.get(URL, headers=UA, timeout=40)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))
    df.to_csv(CACHE, index=False, encoding="utf-8")
    return df


# ---------- retirar a margem da casa ----------
def devig_simples(o):
    """Normaliza pela soma. E o que todo mundo faz, e e enviesado: distorce
    sistematicamente favoritos contra zebras (vies favorito-zebra)."""
    p = 1.0 / o
    return p / p.sum(axis=1, keepdims=True)


def devig_shin(o):
    """Metodo de Shin (1992). Modela a margem como protecao do bookmaker contra
    apostador informado, e por isso corrige o vies favorito-zebra em vez de
    espalhar a margem igualmente por todos os resultados.

    A formula usa a probabilidade implicada BRUTA (1/odd) e a soma do livro B,
    nao o valor ja normalizado. Passar o normalizado faz a funcao virar um
    no-op silencioso: a raiz deixa de existir no intervalo e z cai em zero
    sempre, devolvendo exatamente o de-vig ingenuo. Foi o que aconteceu na
    primeira versao, e o teste ao lado existe para isso nao repetir."""
    bruta = 1.0 / o
    B = bruta.sum(axis=1, keepdims=True)
    out = np.empty_like(bruta)
    for i in range(len(bruta)):
        pi, Bi = bruta[i], B[i, 0]

        def probs(z):
            return (np.sqrt(z * z + 4 * (1 - z) * pi * pi / Bi) - z) / (2 * (1 - z))

        def soma(z):
            return probs(z).sum() - 1.0

        try:
            z = brentq(soma, 1e-12, 0.5)
        except ValueError:
            z = 0.0
        v = probs(z) if z > 0 else pi / Bi
        out[i] = v / v.sum()
    return out


def logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


def main():
    df = baixar("--refresh" in sys.argv)
    df = df[df.HG.notna() & df.AG.notna()].copy()
    df["y"] = np.where(df.HG > df.AG, 0, np.where(df.HG == df.AG, 1, 2))
    # data para ordenar dentro da temporada
    df["dt"] = pd.to_datetime(df.Date, dayfirst=True, errors="coerce")

    fontes = {"media do mercado": ("AvgCH", "AvgCD", "AvgCA"),
              "Betfair Exchange": ("BFECH", "BFECD", "BFECA"),
              "Bet365": ("B365CH", "B365CD", "B365CA")}

    print(f"{len(df)} jogos com placar, temporadas "
          f"{int(df.Season.min())}–{int(df.Season.max())}")
    for nome, cols in fontes.items():
        print(f"  {nome:20s} odds em {df[list(cols)].notna().all(1).mean()*100:5.1f}% dos jogos")

    reg = []
    for temporada, g in df.groupby("Season"):
        g = g.sort_values("dt").reset_index(drop=True)
        if len(g) < MIN_TREINO + BLOCO:
            continue
        times = sorted(set(g.Home) | set(g.Away))
        idx = {t: i for i, t in enumerate(times)}
        H = g.Home.map(idx).to_numpy()
        A = g.Away.map(idx).to_numpy()
        GH = g.HG.to_numpy(int)
        GA = g.AG.to_numpy(int)
        Y = g.y.to_numpy()
        for corte in range(MIN_TREINO, len(g) - BLOCO + 1, BLOCO):
            tr = slice(0, corte)
            te = slice(corte, corte + BLOCO)
            dc = DixonColes(times).fit(H[tr], A[tr], GH[tr], GA[tr])
            pm = dc.probs_1x2(H[te], A[te])
            taxa = np.bincount(Y[tr], minlength=3) / corte
            sub = g.iloc[te]
            linha = {"temporada": int(temporada), "corte": corte,
                     "modelo": logloss(pm, Y[te]),
                     "taxa_liga": logloss(np.tile(taxa, (BLOCO, 1)), Y[te]),
                     "uniforme": np.full(BLOCO, np.log(3))}
            for nome, cols in fontes.items():
                o = sub[list(cols)].to_numpy(float)
                ok = np.isfinite(o).all(1)
                pr = np.full((BLOCO, 3), np.nan)
                if ok.any():
                    pr[ok] = devig_shin(o[ok])
                ll = np.full(BLOCO, np.nan)
                if ok.any():
                    ll[ok] = logloss(pr[ok], Y[te][ok])
                linha[nome] = ll
                if nome == "media do mercado" and ok.any():
                    ps = np.full((BLOCO, 3), np.nan)
                    ps[ok] = devig_simples(o[ok])
                    lls = np.full(BLOCO, np.nan)
                    lls[ok] = logloss(ps[ok], Y[te][ok])
                    linha["media (devig simples)"] = lls
            reg.append(linha)
            print(f"  {int(temporada)} ate o jogo {corte}", end="\r")
    print(" " * 40)

    cols = ["modelo", "media do mercado", "media (devig simples)",
            "Betfair Exchange", "Bet365", "taxa_liga", "uniforme"]
    dados = {c: np.concatenate([r[c] for r in reg if c in r]) for c in cols}
    base = dados["modelo"]
    res = []
    for c in cols:
        v = dados[c]
        ok = np.isfinite(v) & np.isfinite(base)
        d = base[ok] - v[ok]
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else 0
        res.append({"preditor": c, "n": int(ok.sum()),
                    "log_loss": round(float(np.nanmean(v)), 4),
                    "vs_modelo": round(float(d.mean()), 4),
                    "t": round(float(t), 2)})
    tab = pd.DataFrame(res).sort_values("log_loss")
    print(f"LOG-LOSS 1X2, caminhada para a frente ({len(base)} previsoes)")
    print("vs_modelo: quanto o preditor ganha do Dixon-Coles, em nats/jogo\n")
    print(tab.to_string(index=False))

    mer = dados["media do mercado"]
    ok = np.isfinite(mer) & np.isfinite(base)
    piso = np.nanmean(mer[ok])
    teto = np.log(3)
    meu = np.nanmean(base[ok])
    liga = np.nanmean(dados["taxa_liga"][ok])
    print(f"\n--- onde o modelo esta na regua ---")
    print(f"  chutar 1/3 (nada)         {teto:.4f}")
    print(f"  taxa media da liga        {liga:.4f}")
    print(f"  ESTE MODELO               {meu:.4f}")
    print(f"  mercado (Shin)            {piso:.4f}")
    frac = (liga - meu) / (liga - piso) * 100 if liga > piso else float("nan")
    print(f"\n  do caminho entre o baseline e o mercado, o modelo andou "
          f"{frac:.0f}%")
    print(f"  falta para o mercado: {meu - piso:.4f} nats/jogo")

    tab.to_csv("out/regua_mercado.csv", index=False, encoding="utf-8")
    print("\ngravado out/regua_mercado.csv")


if __name__ == "__main__":
    main()
