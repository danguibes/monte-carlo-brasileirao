"""Dixon-Coles: forca de ataque e defesa por time, a partir dos jogos ja disputados.

lambda_mandante = exp(ataque[casa] + defesa[fora] + vantagem_casa)
lambda_visitante = exp(ataque[fora] + defesa[casa])

com a correcao tau de Dixon & Coles (1997) para a dependencia observada em
placares baixos (0-0, 1-0, 0-1, 1-1), que a Poisson independente erra.
Restricoes de identificacao: soma(ataque) = 0 e soma(defesa) = 0.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAXG = 10  # teto de gols por time na grade de placares


def tau(gh, ga, lh, la, rho):
    """Correcao de dependencia nos quatro placares baixos."""
    t = np.ones_like(lh, dtype=float)
    m00 = (gh == 0) & (ga == 0)
    m01 = (gh == 0) & (ga == 1)
    m10 = (gh == 1) & (ga == 0)
    m11 = (gh == 1) & (ga == 1)
    t = np.where(m00, 1 - lh * la * rho, t)
    t = np.where(m01, 1 + lh * rho, t)
    t = np.where(m10, 1 + la * rho, t)
    t = np.where(m11, 1 - rho, t)
    return t


class DixonColes:
    def __init__(self, teams):
        self.teams = list(teams)
        self.n = len(teams)
        self.idx = {t: i for i, t in enumerate(teams)}

    def _unpack(self, p):
        n = self.n
        att = np.append(p[:n - 1], -p[:n - 1].sum())
        dfn = np.append(p[n - 1:2 * n - 2], -p[n - 1:2 * n - 2].sum())
        return att, dfn, p[-2], p[-1]  # att, def, home_adv, rho

    def _lams(self, p, h, a):
        att, dfn, home, _ = self._unpack(p)
        return (np.exp(att[h] + dfn[a] + home), np.exp(att[a] + dfn[h]))

    def fit(self, h, a, gh, ga, w=None):
        """h, a: indices dos times. gh, ga: gols. w: peso por jogo (recencia)."""
        w = np.ones(len(h)) if w is None else np.asarray(w, float)
        n = self.n
        p0 = np.concatenate([np.zeros(n - 1), np.zeros(n - 1), [0.25], [0.0]])

        def nll(p):
            if not (-0.2 < p[-1] < 0.2):
                return 1e9
            lh, la = self._lams(p, h, a)
            t = tau(gh, ga, lh, la, p[-1])
            if np.any(t <= 0):
                return 1e9
            ll = (np.log(t) + poisson.logpmf(gh, lh) + poisson.logpmf(ga, la))
            return -np.sum(w * ll)

        res = minimize(nll, p0, method="L-BFGS-B",
                       options={"maxiter": 20000, "maxfun": 100000})
        self.params = res.x
        self.res = res
        att, dfn, self.home_adv, self.rho = self._unpack(res.x)
        self.attack = pd.Series(att, index=self.teams)
        self.defence = pd.Series(dfn, index=self.teams)
        return self

    def lams(self, h, a):
        return self._lams(self.params, h, a)

    def score_grid(self, h, a):
        """Distribuicao conjunta de placares (M, MAXG+1, MAXG+1), normalizada."""
        lh, la = self.lams(h, a)
        g = np.arange(MAXG + 1)
        ph = poisson.pmf(g[None, :], lh[:, None])          # (M, G)
        pa = poisson.pmf(g[None, :], la[:, None])
        joint = ph[:, :, None] * pa[:, None, :]            # (M, G, G)
        GH, GA = np.meshgrid(g, g, indexing="ij")
        t = tau(GH[None], GA[None],
                lh[:, None, None], la[:, None, None], self.rho)
        joint = joint * t
        return joint / joint.sum(axis=(1, 2), keepdims=True)

    def probs_1x2(self, h, a):
        j = self.score_grid(h, a)
        GH, GA = np.meshgrid(np.arange(MAXG + 1), np.arange(MAXG + 1), indexing="ij")
        return np.stack([j[:, GH > GA].sum(1), j[:, GH == GA].sum(1), j[:, GH < GA].sum(1)], 1)


def recency_weights(n_played, half_life=None):
    """Peso exponencial por ordem do jogo. Sem datas na fonte, o indice do jogo
    e usado como proxy grosseiro; half_life=None devolve peso uniforme."""
    if half_life is None:
        return np.ones(n_played)
    age = n_played - 1 - np.arange(n_played)
    return 0.5 ** (age / half_life)
