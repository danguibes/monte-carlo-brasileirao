"""O modelo esta achatando as diferencas entre os times?

Tres perguntas, tres medicoes:

  A. O placar mais provavel e mesmo 1-1 quase sempre? E isso quer dizer o que?
  B. A taxa de empate do modelo bate com a observada?
  C. TESTE DECISIVO — dispersao. Simula os jogos JA DISPUTADOS com o modelo
     ajustado e compara o espalhamento da tabela resultante com o real. Se o
     modelo achata forcas, a tabela simulada fica mais apertada que a de
     verdade, e isso aparece aqui.
"""
import numpy as np
import pandas as pd
from model import MAXG
from run import load, fit_model, tabela

N = 3000


def main():
    matches, st = load()
    dc, (h, a, gh, ga) = fit_model(matches, None)
    teams = dc.teams

    # ---------- A. o placar modal ----------
    rem = matches[~matches.played].reset_index(drop=True)
    hi = rem.home.map(dc.idx).to_numpy()
    ai = rem.away.map(dc.idx).to_numpy()
    grid = dc.score_grid(hi, ai)
    g = np.arange(MAXG + 1)
    GH, GA = np.meshgrid(g, g, indexing="ij")
    flat = grid.reshape(len(hi), -1)
    k = flat.argmax(1)
    modal = pd.Series([f"{i // (MAXG+1)}-{i % (MAXG+1)}" for i in k]).value_counts()
    pmodal = flat.max(1)
    p1x2 = dc.probs_1x2(hi, ai)

    print("A. PLACAR MAIS PROVAVEL nos", len(hi), "jogos que faltam")
    print(modal.head(6).to_string())
    print(f"\n   probabilidade media DO placar modal: {pmodal.mean()*100:.1f}%")
    print(f"   ou seja, o placar 'mais provavel' e improvavel: erra {100-pmodal.mean()*100:.0f}% das vezes.")
    print(f"   sao {(MAXG+1)**2} placares possiveis; a moda de uma distribuicao")
    print("   espalhada em 81 celulas nao resume quase nada.\n")

    emp = p1x2[:, 1]
    print(f"   P(empate) media do modelo nesses jogos: {emp.mean()*100:.1f}%")
    print(f"   P(vitoria do mandante) media:           {p1x2[:,0].mean()*100:.1f}%")
    print(f"   jogos em que o EMPATE e o resultado mais provavel: "
          f"{int((p1x2.argmax(1) == 1).sum())} de {len(hi)}")

    # ---------- B. taxa de empate ----------
    obs_e = (gh == ga).mean()
    mod_e = dc.probs_1x2(h, a)[:, 1].mean()
    print(f"\nB. EMPATES  observado {obs_e*100:.1f}%   modelo {mod_e*100:.1f}%   "
          f"(diferenca {abs(obs_e-mod_e)*100:.1f} p.p.)")
    print(f"   rho = {dc.rho:+.3f}: a correcao de Dixon-Coles AUMENTA o 1-1 em "
          f"{-dc.rho*100:.1f}%, de proposito,")
    print("   porque o futebol real tem mais 1-1 do que a Poisson independente preve.")

    # ---------- C. dispersao ----------
    real = tabela(matches).set_index("team").loc[teams]
    rng = np.random.default_rng(3)
    gridp = dc.score_grid(h, a)
    flatp = gridp.reshape(len(h), -1)
    cum = flatp.cumsum(1) / flatp.sum(1, keepdims=True)
    sds, amps = [], []
    for _ in range(N):
        pts = np.zeros(len(teams), int)
        u = rng.random(len(h))
        idx = (cum < u[:, None]).sum(1)
        sh, sa = idx // (MAXG + 1), idx % (MAXG + 1)
        for m in range(len(h)):
            if sh[m] > sa[m]:
                pts[h[m]] += 3
            elif sh[m] == sa[m]:
                pts[h[m]] += 1; pts[a[m]] += 1
            else:
                pts[a[m]] += 3
        sds.append(pts.std()); amps.append(pts.max() - pts.min())

    sd_real, amp_real = real.pts.std(), real.pts.max() - real.pts.min()
    sds, amps = np.array(sds), np.array(amps)
    print(f"\nC. DISPERSAO da tabela, refazendo os {len(h)} jogos ja disputados "
          f"({N} vezes)")
    print(f"   desvio-padrao dos pontos   real {sd_real:5.2f}   "
          f"modelo {sds.mean():5.2f} (p5 {np.percentile(sds,5):.2f}, p95 {np.percentile(sds,95):.2f})")
    print(f"   amplitude 1o ao 20o        real {amp_real:5.0f}   "
          f"modelo {amps.mean():5.1f} (p5 {np.percentile(amps,5):.0f}, p95 {np.percentile(amps,95):.0f})")
    pct_sd = (sds >= sd_real).mean() * 100
    print(f"\n   a tabela real e MAIS espalhada que {100-pct_sd:.0f}% das simuladas.")
    if 5 < pct_sd < 95:
        print("   -> o modelo reproduz o espalhamento observado. Nao esta achatando.")
    elif pct_sd <= 5:
        print("   -> o modelo ACHATA: a tabela real e mais espalhada do que ele consegue gerar.")
    else:
        print("   -> o modelo EXAGERA as diferencas.")

    # quanto da tabela e forca e quanto e sorte
    # ---------- D. quanto da tabela e forca e quanto e sorte ----------
    # Refaz a temporada com os 20 times IDENTICOS (so a vantagem de casa).
    # A variancia que sobra ai e puro acaso; a diferenca para a real e forca.
    med = np.exp(dc.attack.mean() + dc.defence.mean())
    lh0 = np.full(len(h), np.exp(dc.home_adv) * med)
    la0 = np.full(len(h), med)
    sds0 = []
    for _ in range(N):
        sh = rng.poisson(lh0)
        sa = rng.poisson(la0)
        pts = np.zeros(len(teams), int)
        np.add.at(pts, h, np.where(sh > sa, 3, np.where(sh == sa, 1, 0)))
        np.add.at(pts, a, np.where(sa > sh, 3, np.where(sh == sa, 1, 0)))
        sds0.append(pts.std())
    v_acaso = float(np.mean(np.array(sds0) ** 2))
    print("\nD. QUANTO DA TABELA E FORCA, QUANTO E SORTE")
    print(f"   variancia dos pontos na tabela real       : {sd_real**2:6.1f}")
    print(f"   variancia com os 20 times IDENTICOS       : {v_acaso:6.1f}   (puro acaso)")
    print(f"   -> parcela atribuivel a forca de verdade  : "
          f"{max(0.0, 1 - v_acaso / sd_real**2)*100:5.0f}%")

    forcas = pd.DataFrame({"ataque": dc.attack, "defesa": dc.defence})
    ind = (forcas.ataque - forcas.defesa).sort_values(ascending=False)
    print(f"\n   amplitude de forca ataque-defesa: {ind.iloc[0]-ind.iloc[-1]:.2f} em log")
    print(f"   traduzindo: o melhor faz {np.exp(ind.iloc[0]-ind.iloc[-1]):.2f}x "
          "mais gols que o pior, em condicoes iguais")


if __name__ == "__main__":
    main()
