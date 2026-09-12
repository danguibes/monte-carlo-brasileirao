"""A vantagem de casa devia ser por time, e nao uma so para a liga?

A pergunta e boa a priori: altitude, distancia de viagem, tamanho da torcida e
gramado sao diferentes em cada estadio. O custo e que vira 20 parametros novos
estimados de ~13 jogos em casa cada um — territorio classico de ruido virando
sinal.

O teste decisivo e o mesmo padrao do diagnostico: simular temporadas a partir
do modelo de UMA vantagem so, ajustar vantagens POR TIME nessas temporadas
inventadas, e ver quanto espalhamento o puro acaso produz. Se o espalhamento
observado nao passar do que o acaso faz, os 20 parametros nao estao medindo
nada.

    python fator_casa.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from model import DixonColes, tau, MAXG
from run import load, fit_model

B = 200          # temporadas sinteticas para a distribuicao nula
REP, TEST = 12, 0.25   # validacao fora da amostra


class CasaPorTime(DixonColes):
    """Igual ao Dixon-Coles, mas com uma vantagem de casa por mandante.
    Identificacao: a media das vantagens e o parametro global."""

    def _unpack(self, p):
        n = self.n
        att = np.append(p[:n-1], -p[:n-1].sum())
        dfn = np.append(p[n-1:2*n-2], -p[n-1:2*n-2].sum())
        casa = p[2*n-2:3*n-2]          # uma por time, sem restricao de soma
        return att, dfn, casa, p[-1]

    def _lams(self, p, h, a):
        att, dfn, casa, _ = self._unpack(p)
        return np.exp(att[h] + dfn[a] + casa[h]), np.exp(att[a] + dfn[h])

    def fit(self, h, a, gh, ga, w=None):
        w = np.ones(len(h)) if w is None else np.asarray(w, float)
        n = self.n
        p0 = np.concatenate([np.zeros(n-1), np.zeros(n-1), np.full(n, 0.25), [0.0]])

        def nll(p):
            if not (-0.2 < p[-1] < 0.2):
                return 1e9
            lh, la = self._lams(p, h, a)
            t = tau(gh, ga, lh, la, p[-1])
            if np.any(t <= 0):
                return 1e9
            return -np.sum(w * (np.log(t) + poisson.logpmf(gh, lh)
                                + poisson.logpmf(ga, la)))

        res = minimize(nll, p0, method="L-BFGS-B",
                       options={"maxiter": 40000, "maxfun": 200000})
        self.params = res.x
        self.res = res
        att, dfn, casa, self.rho = self._unpack(res.x)
        self.attack = pd.Series(att, index=self.teams)
        self.defence = pd.Series(dfn, index=self.teams)
        self.casa = pd.Series(casa, index=self.teams)
        self.home_adv = float(casa.mean())
        return self


def logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


def sortear(dc, h, a, rng):
    g = dc.score_grid(h, a).reshape(len(h), -1)
    cum = g.cumsum(1) / g.sum(1, keepdims=True)
    idx = (cum < rng.random(len(h))[:, None]).sum(1)
    return idx // (MAXG + 1), idx % (MAXG + 1)


def main():
    matches, st = load()
    dc, (h, a, gh, ga) = fit_model(matches, None)
    teams = dc.teams
    Y = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))

    pt = CasaPorTime(teams).fit(h, a, gh, ga)
    obs = pt.casa.std()
    print(f"vantagem de casa unica       : {dc.home_adv:+.3f} log "
          f"({np.exp(dc.home_adv):.3f}x)")
    print(f"vantagem por time, media     : {pt.casa.mean():+.3f} log")
    print(f"          desvio entre times : {obs:.3f} log")
    print(f"          min .. max         : {pt.casa.min():+.3f} .. {pt.casa.max():+.3f}")
    print("\nmaiores e menores:")
    ord_ = pt.casa.sort_values(ascending=False)
    print(pd.concat([ord_.head(4), ord_.tail(4)]).round(3).to_string())

    # ---------- distribuicao nula: quanto o acaso espalha? ----------
    rng = np.random.default_rng(31)
    nulos = []
    for b in range(B):
        sh, sa = sortear(dc, h, a, rng)       # mundo com UMA vantagem so
        nulos.append(CasaPorTime(teams).fit(h, a, sh, sa).casa.std())
        print(f"  nulo {b+1}/{B}", end="\r")
    nulos = np.array(nulos)
    p = (nulos >= obs).mean()
    print(" " * 30)
    print("TESTE: num mundo onde a vantagem de casa e IGUAL para todos,")
    print(f"  o desvio estimado entre times daria {nulos.mean():.3f} "
          f"(p5 {np.percentile(nulos,5):.3f}, p95 {np.percentile(nulos,95):.3f})")
    print(f"  observado: {obs:.3f}")
    print(f"  p-valor: {p:.3f}  ->  ", end="")
    print("ha diferenca real entre os times" if p < 0.05
          else "NAO da para distinguir de ruido")

    # ---------- e fora da amostra, ajuda? ----------
    rng2 = np.random.default_rng(7)
    d_uni, d_pt = [], []
    for _ in range(REP):
        idx = rng2.permutation(len(h))
        nt = int(len(h) * TEST)
        te, tr = idx[:nt], idx[nt:]
        m1 = DixonColes(teams).fit(h[tr], a[tr], gh[tr], ga[tr])
        m2 = CasaPorTime(teams).fit(h[tr], a[tr], gh[tr], ga[tr])
        d_uni += list(logloss(m1.probs_1x2(h[te], a[te]), Y[te]))
        d_pt += list(logloss(m2.probs_1x2(h[te], a[te]), Y[te]))
    d_uni, d_pt = np.array(d_uni), np.array(d_pt)
    dif = d_uni - d_pt
    t = dif.mean() / (dif.std(ddof=1) / np.sqrt(len(dif)))
    print(f"\nfora da amostra ({len(d_uni)} previsoes):")
    print(f"  vantagem unica    {d_uni.mean():.4f} nats")
    print(f"  vantagem por time {d_pt.mean():.4f} nats")
    print(f"  ganho de 20 parametros a mais: {dif.mean():+.4f} nats/jogo (t={t:.2f})")

    pd.DataFrame({"time": teams, "casa_log": pt.casa[teams].round(3).to_numpy(),
                  "casa_x": np.exp(pt.casa[teams]).round(3)}).to_csv(
        "out/fator_casa.csv", index=False, encoding="utf-8")
    print("\ngravado out/fator_casa.csv")


if __name__ == "__main__":
    main()
