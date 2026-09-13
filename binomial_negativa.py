"""A Poisson serve para gols de futebol, ou o certo seria binomial negativa?

A Poisson impõe uma restricao forte e nao negociada: variancia IGUAL a media.
Se o futebol tiver mais dispersao que isso — goleadas mais frequentes do que a
Poisson admite — a grade de placares subestima as pontas, e o saldo de gols
simulado fica apertado demais.

A binomial negativa relaxa exatamente isso: Var = mu + mu^2/r. Com r grande ela
VIRA a Poisson, entao o teste e limpo — se r estimado for enorme e o ganho de
verossimilhanca nao pagar o parametro a mais, a Poisson esta absolvida.

Quatro medidas:
  A. dispersao de Pearson (1.0 = Poisson perfeita)
  B. observado x esperado por numero de gols
  C. verossimilhanca e AIC, Poisson contra binomial negativa
  D. log-loss 1X2 e do PLACAR EXATO, fora da amostra

    python binomial_negativa.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson, nbinom
from model import DixonColes, tau, MAXG
from run import load, fit_model

REP, TEST = 10, 0.25


def nb_logpmf(k, mu, r):
    """log P(K = k) com media mu e parametro de forma r. r -> inf vira Poisson."""
    k = np.asarray(k, float)
    return (gammaln(k + r) - gammaln(r) - gammaln(k + 1)
            + r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu)))


class DixonColesNB(DixonColes):
    """Mesma estrutura de ataque/defesa/casa, marginais binomiais negativas."""

    def _unpack(self, p):
        n = self.n
        att = np.append(p[:n-1], -p[:n-1].sum())
        dfn = np.append(p[n-1:2*n-2], -p[n-1:2*n-2].sum())
        return att, dfn, p[-3], p[-2], p[-1]   # att, def, casa, rho, log_r

    def _lams(self, p, h, a):
        att, dfn, casa, _, _ = self._unpack(p)
        return np.exp(att[h] + dfn[a] + casa), np.exp(att[a] + dfn[h])

    def fit(self, h, a, gh, ga, w=None):
        w = np.ones(len(h)) if w is None else np.asarray(w, float)
        n = self.n
        p0 = np.concatenate([np.zeros(n-1), np.zeros(n-1), [0.25], [0.0], [3.0]])

        def nll(p):
            if not (-0.2 < p[-2] < 0.2) or not (-2 < p[-1] < 12):
                return 1e9
            r = np.exp(p[-1])
            lh, la = self._lams(p, h, a)
            t = tau(gh, ga, lh, la, p[-2])
            if np.any(t <= 0):
                return 1e9
            return -np.sum(w * (np.log(t) + nb_logpmf(gh, lh, r) + nb_logpmf(ga, la, r)))

        res = minimize(nll, p0, method="L-BFGS-B",
                       options={"maxiter": 40000, "maxfun": 200000})
        self.params, self.res = res.x, res
        att, dfn, self.home_adv, self.rho, log_r = self._unpack(res.x)
        self.r = float(np.exp(log_r))
        self.attack = pd.Series(att, index=self.teams)
        self.defence = pd.Series(dfn, index=self.teams)
        return self

    def score_grid(self, h, a):
        lh, la = self.lams(h, a)
        g = np.arange(MAXG + 1)
        ph = np.exp(nb_logpmf(g[None, :], lh[:, None], self.r))
        pa = np.exp(nb_logpmf(g[None, :], la[:, None], self.r))
        joint = ph[:, :, None] * pa[:, None, :]
        GH, GA = np.meshgrid(g, g, indexing="ij")
        joint = joint * tau(GH[None], GA[None], lh[:, None, None],
                            la[:, None, None], self.rho)
        return joint / joint.sum(axis=(1, 2), keepdims=True)


def logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


def main():
    matches, st = load()
    dc, (h, a, gh, ga) = fit_model(matches, None)
    teams = dc.teams
    Y = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))

    # cada jogo da DUAS observacoes de gol: mandante e visitante
    lh, la = dc.lams(h, a)
    y = np.concatenate([gh, ga])
    mu = np.concatenate([lh, la])
    k = 2 * dc.n - 2 + 2                       # parametros do modelo Poisson

    # ---------- A. dispersao ----------
    pearson = np.sum((y - mu) ** 2 / mu) / (len(y) - k)
    print(f"A. DISPERSAO DE PEARSON: {pearson:.3f}")
    print(f"   (1,00 = Poisson perfeita; acima disso sobra variancia)")
    print(f"   media dos gols {y.mean():.3f}  variancia {y.var(ddof=1):.3f}")

    # Pearson in-sample e enviesado para baixo: mu vem de um modelo com 40+
    # parametros ajustado nesses mesmos jogos, entao os residuos encolhem.
    # Refaz FORA DA AMOSTRA, com mu estimado so no treino.
    rng0 = np.random.default_rng(19)
    disp = []
    for _ in range(20):
        idx = rng0.permutation(len(h))
        nt = int(len(h) * TEST)
        te, tr = idx[:nt], idx[nt:]
        mt = DixonColes(teams).fit(h[tr], a[tr], gh[tr], ga[tr])
        l1, l2 = mt.lams(h[te], a[te])
        yy = np.concatenate([gh[te], ga[te]])
        mm = np.concatenate([l1, l2])
        disp.append(np.sum((yy - mm) ** 2 / mm) / len(yy))
    disp = np.array(disp)
    print(f"   fora da amostra: {disp.mean():.3f} "
          f"(p5 {np.percentile(disp,5):.3f}, p95 {np.percentile(disp,95):.3f}, 20 particoes)")

    # ---------- B. observado x esperado ----------
    print("\nB. DISTRIBUICAO DE GOLS POR TIME EM CADA JOGO")
    print("   gols   observado   Poisson   dif")
    linhas = []
    for g in range(6):
        obs = (y == g).mean() * 100
        esp = poisson.pmf(g, mu).mean() * 100
        linhas.append((g, obs, esp))
        print(f"   {g}      {obs:6.2f}%    {esp:6.2f}%   {obs-esp:+5.2f}")
    obs5 = (y >= 6).mean() * 100
    esp5 = (1 - poisson.cdf(5, mu)).mean() * 100
    print(f"   6+     {obs5:6.2f}%    {esp5:6.2f}%   {obs5-esp5:+5.2f}")

    # ---------- C. verossimilhanca ----------
    nb = DixonColesNB(teams).fit(h, a, gh, ga)
    ll_p = -dc.res.fun
    ll_nb = -nb.res.fun
    aic_p = 2 * k - 2 * ll_p
    aic_nb = 2 * (k + 1) - 2 * ll_nb
    print(f"\nC. AJUSTE (mesmos {len(h)} jogos)")
    print(f"   Poisson             log-verossimilhanca {ll_p:9.2f}   AIC {aic_p:8.2f}")
    print(f"   binomial negativa   log-verossimilhanca {ll_nb:9.2f}   AIC {aic_nb:8.2f}")
    print(f"   r estimado: {nb.r:.1f}   (r grande = indistinguivel da Poisson)")
    print(f"   variancia extra no lambda medio: "
          f"{mu.mean()**2 / nb.r:.4f} gol^2 sobre uma media de {mu.mean():.3f}")
    razao = 2 * (ll_nb - ll_p)
    print(f"   teste da razao de verossimilhanca: {razao:.3f} "
          f"(qui-quadrado 1 gl; 3,84 = 5%)  -> ", end="")
    print("binomial negativa ganha" if razao > 3.84 else "NAO justifica o parametro extra")

    # ---------- D. fora da amostra ----------
    rng = np.random.default_rng(5)
    l1x2_p, l1x2_nb, lex_p, lex_nb = [], [], [], []
    for _ in range(REP):
        idx = rng.permutation(len(h))
        nt = int(len(h) * TEST)
        te, tr = idx[:nt], idx[nt:]
        mp = DixonColes(teams).fit(h[tr], a[tr], gh[tr], ga[tr])
        mn = DixonColesNB(teams).fit(h[tr], a[tr], gh[tr], ga[tr])
        l1x2_p += list(logloss(mp.probs_1x2(h[te], a[te]), Y[te]))
        l1x2_nb += list(logloss(mn.probs_1x2(h[te], a[te]), Y[te]))
        for mod, dest in ((mp, lex_p), (mn, lex_nb)):
            gr = mod.score_grid(h[te], a[te])
            gg = np.clip(gh[te], 0, MAXG), np.clip(ga[te], 0, MAXG)
            dest += list(-np.log(np.clip(gr[np.arange(nt), gg[0], gg[1]], 1e-12, None)))
    for nome, p_, n_ in (("1X2", l1x2_p, l1x2_nb), ("placar exato", lex_p, lex_nb)):
        p_, n_ = np.array(p_), np.array(n_)
        d = p_ - n_
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        print(f"\nD. LOG-LOSS FORA DA AMOSTRA — {nome} ({len(p_)} previsoes)")
        print(f"   Poisson            {p_.mean():.4f}")
        print(f"   binomial negativa  {n_.mean():.4f}")
        print(f"   ganho da NB: {d.mean():+.4f} nats/previsao (t={t:.2f})")

    pd.DataFrame(linhas, columns=["gols", "observado_%", "poisson_%"]).to_csv(
        "out/dispersao_gols.csv", index=False, encoding="utf-8")
    print("\ngravado out/dispersao_gols.csv")


if __name__ == "__main__":
    main()
