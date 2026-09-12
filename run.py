"""Monte Carlo do Campeonato Brasileiro Serie A.

  python run.py --n 50000
  python run.py --n 50000 --half-life 40
  python run.py --n 50000 --dado "Flamengo>Palmeiras"     # condicional
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from model import DixonColes, MAXG, recency_weights

# Criterios de desempate da CBF, na ordem: pontos, vitorias, saldo, gols pro.
# Do 5o em diante (confronto direto, cartoes) o desempate e sorteado.
G4, G6, Z4 = 4, 6, 4


def tabela(matches):
    """Classificacao derivada dos proprios jogos disputados."""
    ag = {}
    for r in matches[matches.played].itertuples():
        for team, gf, ga in ((r.home, r.home_goals, r.away_goals),
                             (r.away, r.away_goals, r.home_goals)):
            a = ag.setdefault(team, dict(j=0, v=0, e=0, d=0, gp=0, gc=0))
            a["j"] += 1; a["gp"] += int(gf); a["gc"] += int(ga)
            a["v" if gf > ga else "e" if gf == ga else "d"] += 1
    st = pd.DataFrame(ag).T.reset_index(names="team")
    st["pts"] = st.v * 3 + st.e
    st["sg"] = st.gp - st.gc
    return st.sort_values(["pts", "v", "sg", "gp"], ascending=False).reset_index(drop=True)


def load():
    """A classificacao sai dos jogos, nao de uma tabela baixada. Se houver uma
    tabela publicada em data/standings.csv, ela vira CONFERENCIA: divergencia
    costuma significar punicao em pontos, que os jogos nao mostram."""
    m = pd.read_csv("data/matches.csv", encoding="utf-8")
    st = tabela(m)
    pub = "data/standings.csv"
    if os.path.exists(pub):
        o = pd.read_csv(pub, encoding="utf-8").set_index("team")
        # So compara se as duas fontes sao do MESMO momento. Se a tabela
        # publicada tem outro total de jogos, ela e de outra busca, e a
        # divergencia seria so a diferenca de horario — alarme falso.
        if int(o.j.sum()) != int(st.j.sum()):
            print(f"(tabela publicada e de outro momento: {int(o.j.sum())//2} jogos "
                  f"contra {int(st.j.sum())//2}; conferencia pulada)")
            return m, st
        j = st.set_index("team").join(o[["pts", "j", "gp", "gc"]], rsuffix="_pub")
        bad = j[(j.pts != j.pts_pub) | (j.j != j.j_pub)
                | (j.gp != j.gp_pub) | (j.gc != j.gc_pub)].dropna(how="all")
        if len(bad):
            print("AVISO: classificacao derivada difere da publicada "
                  f"em {len(bad)} time(s).")
            print("       causas comuns: jogo EM ANDAMENTO (a Wikipedia conta o "
                  "parcial e nos nao) ou punicao em pontos.")
            print(bad[["pts", "pts_pub", "j", "j_pub"]].to_string())
    return m, st


def fit_model(matches, half_life):
    teams = sorted(matches.home.unique())
    dc = DixonColes(teams)
    if half_life is not None and "data" not in matches.columns:
        sys.exit("--half-life exige data por jogo. Rode fetch_globo.py primeiro; "
                 "a matriz da Wikipedia sozinha nao tem datas e o peso sairia "
                 "por ordem alfabetica, que nao e ordem nenhuma.")
    p = matches[matches.played].reset_index(drop=True)
    if half_life is not None:
        p = p.sort_values("data").reset_index(drop=True)
    h = p.home.map(dc.idx).to_numpy()
    a = p.away.map(dc.idx).to_numpy()
    gh = p.home_goals.to_numpy(int)
    ga = p.away_goals.to_numpy(int)
    dc.fit(h, a, gh, ga, w=recency_weights(len(p), half_life))
    return dc, (h, a, gh, ga)


def apply_h2h(dc, hi, ai, weight, k_shrink=10.0):
    """Ajuste opcional por confronto direto historico (data/h2h.csv).

    Desligado por padrao: com ~2 jogos por par por temporada, o sinal e
    dominado por ruido. Exige arquivo historico proprio; nada e inventado.
    """
    path = "data/h2h.csv"
    lh, la = dc.lams(hi, ai)
    if weight <= 0:
        return lh, la
    if not os.path.exists(path):
        sys.exit("--h2h-weight > 0 exige data/h2h.csv com colunas "
                 "home,away,home_goals,away_goals (temporadas anteriores).")
    hist = pd.read_csv(path, encoding="utf-8")
    inv = {v: k for k, v in dc.idx.items()}
    for m in range(len(hi)):
        sub = hist[(hist.home == inv[hi[m]]) & (hist.away == inv[ai[m]])]
        k = len(sub)
        if k == 0:
            continue
        alpha = weight * k / (k + k_shrink)
        lh[m] = np.exp((1 - alpha) * np.log(lh[m]) + alpha * np.log(max(sub.home_goals.mean(), .2)))
        la[m] = np.exp((1 - alpha) * np.log(la[m]) + alpha * np.log(max(sub.away_goals.mean(), .2)))
    return lh, la


def simulate(dc, rem, base, n, seed=1, forced=None):
    """rem: jogos restantes. base: pts/v/gp/gc atuais por time.

    forced: dict {indice_do_jogo: "H"|"D"|"A"} para probabilidade condicional.
    """
    rng = np.random.default_rng(seed)
    hi = rem.home.map(dc.idx).to_numpy()
    ai = rem.away.map(dc.idx).to_numpy()
    grid = dc.score_grid(hi, ai)
    g = np.arange(MAXG + 1)
    GH, GA = np.meshgrid(g, g, indexing="ij")
    flatGH, flatGA = GH.ravel(), GA.ravel()

    nt = dc.n
    pts = np.tile(base["pts"], (n, 1)).astype(np.int32)
    win = np.tile(base["v"], (n, 1)).astype(np.int32)
    gp = np.tile(base["gp"], (n, 1)).astype(np.int32)
    gc = np.tile(base["gc"], (n, 1)).astype(np.int32)

    for m in range(len(hi)):
        w = grid[m].ravel().copy()
        if forced and m in forced:
            mask = {"H": flatGH > flatGA, "D": flatGH == flatGA,
                    "A": flatGH < flatGA}[forced[m]]
            w = np.where(mask, w, 0.0)
        w = w / w.sum()
        k = rng.choice(w.size, size=n, p=w)
        sh, sa = flatGH[k], flatGA[k]
        h, a = hi[m], ai[m]
        gp[:, h] += sh
        gc[:, h] += sa
        gp[:, a] += sa
        gc[:, a] += sh
        hw, dr, aw = sh > sa, sh == sa, sh < sa
        pts[:, h] += np.where(hw, 3, np.where(dr, 1, 0))
        pts[:, a] += np.where(aw, 3, np.where(dr, 1, 0))
        win[:, h] += hw
        win[:, a] += aw

    sg = gp - gc
    key = (pts.astype(np.int64) * 10**12 + win.astype(np.int64) * 10**9
           + (sg.astype(np.int64) + 500) * 10**5 + gp.astype(np.int64) * 10
           + rng.integers(0, 10, size=(n, nt)))
    order = np.argsort(-key, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.tile(np.arange(nt), (n, 1)), axis=1)
    return dict(pts=pts, rank=rank + 1, sg=sg, gp=gp, win=win)


def report(dc, sim, st):
    teams = dc.teams
    rank, pts = sim["rank"], sim["pts"]
    nt = len(teams)
    out = pd.DataFrame({
        "time": teams,
        "pts_hoje": st.set_index("team").loc[teams, "pts"].to_numpy(),
        "pts_medio": pts.mean(0).round(1),
        "pts_p10": np.percentile(pts, 10, axis=0).round(0).astype(int),
        "pts_p90": np.percentile(pts, 90, axis=0).round(0).astype(int),
        "pos_mediana": np.median(rank, axis=0).astype(int),
        "titulo_%": (rank == 1).mean(0) * 100,
        "G4_%": (rank <= G4).mean(0) * 100,
        "G6_%": (rank <= G6).mean(0) * 100,
        "Z4_%": (rank >= nt - Z4 + 1).mean(0) * 100,
    }).sort_values(["titulo_%", "pts_medio"], ascending=False)
    for c in ["titulo_%", "G4_%", "G6_%", "Z4_%"]:
        out[c] = out[c].round(2)
    dist = np.stack([[(rank[:, i] == p).mean() * 100 for p in range(1, nt + 1)]
                     for i in range(nt)])
    dist = pd.DataFrame(dist.round(2), index=teams,
                        columns=[f"{p}o" for p in range(1, nt + 1)])
    return out.reset_index(drop=True), dist.loc[out.time]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--half-life", type=float, default=None,
                    help="meia-vida em jogos para peso de recencia (padrao: uniforme)")
    ap.add_argument("--h2h-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dado", action="append", default=[],
                    help='condicional: "Flamengo>Palmeiras" mandante vence, '
                         '"Flamengo=Palmeiras" empate, "Flamengo<Palmeiras" visitante vence')
    args = ap.parse_args()

    matches, st = load()
    dc, (h, a, gh, ga) = fit_model(matches, args.half_life)
    rem = matches[~matches.played].reset_index(drop=True)

    # --- controle: o modelo reproduz o que foi observado? ---
    p1x2 = dc.probs_1x2(h, a)
    obs = np.array([(gh > ga).mean(), (gh == ga).mean(), (gh < ga).mean()]) * 100
    grid = dc.score_grid(h, a)
    g = np.arange(MAXG + 1)
    eg = (grid.sum(2) * g).sum(1) + (grid.sum(1) * g).sum(1)   # esperanca sob tau
    print(f"jogos disputados {len(h)} | restantes {len(rem)}")
    print(f"vantagem de casa exp({dc.home_adv:+.3f}) = {np.exp(dc.home_adv):.3f}x    rho {dc.rho:+.3f}")
    print(f"1X2 observado   M {obs[0]:5.1f}%   E {obs[1]:5.1f}%   V {obs[2]:5.1f}%")
    print(f"1X2 do modelo   M {p1x2[:,0].mean()*100:5.1f}%   E {p1x2[:,1].mean()*100:5.1f}%   V {p1x2[:,2].mean()*100:5.1f}%")
    print(f"gols/jogo observado {(gh.sum()+ga.sum())/len(h):.3f}   modelo {eg.mean():.3f}")

    forced = {}
    for d in args.dado:
        for op, code in ((">", "H"), ("=", "D"), ("<", "A")):
            if op in d:
                hh, aa = [s.strip() for s in d.split(op)]
                hits = rem.index[(rem.home == hh) & (rem.away == aa)]
                if not len(hits):
                    sys.exit(f"jogo nao esta entre os restantes: {hh} x {aa}")
                forced[int(hits[0])] = code
                break

    base = {k: st.set_index("team").loc[dc.teams, k].to_numpy()
            for k in ("pts", "v", "gp", "gc")}
    sim = simulate(dc, rem, base, args.n, args.seed, forced or None)
    out, dist = report(dc, sim, st)

    forcas = pd.DataFrame({"ataque": dc.attack, "defesa": dc.defence}).round(3)
    forcas["indice"] = (dc.attack - dc.defence).round(3)
    forcas = forcas.sort_values("indice", ascending=False)

    tag = "_cond" if forced else ""
    out.to_csv(f"out/projecao{tag}.csv", index=False, encoding="utf-8")
    dist.to_csv(f"out/distribuicao_posicao{tag}.csv", encoding="utf-8")
    forcas.to_csv("out/forcas.csv", encoding="utf-8")
    err = 1.96 * np.sqrt(0.25 / args.n) * 100
    print(f"\n--- {args.n:,} temporadas simuladas (erro de Monte Carlo <= {err:.2f} p.p.) ---")
    if forced:
        print("condicional:", ", ".join(args.dado))
    print(out.to_string(index=False))
    print("\nforcas ajustadas (log): ataque alto = marca mais; defesa baixa = sofre menos")
    print(forcas.to_string())


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    main()
