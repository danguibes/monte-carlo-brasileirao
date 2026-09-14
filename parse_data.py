"""Extrai resultados e jogos restantes do Brasileirao Serie A a partir da
matriz de confrontos da Wikipedia (pt). Gera data/matches.csv e data/standings.csv."""
import io, re, sys, json
import requests
import pandas as pd

PAGE = "Campeonato Brasileiro de Futebol de 2026 - Série A"
UA = {"User-Agent": "brasileirao-mc/0.1 (modelo pessoal)"}
SCORE = re.compile(r"^\s*(\d+)\s*[–—\-]\s*(\d+)\s*$")


def fetch_html(cache="data/page.html", refresh=False):
    import os
    if os.path.exists(cache) and not refresh:
        return open(cache, encoding="utf-8").read()
    r = requests.get("https://pt.wikipedia.org/w/api.php", headers=UA, params={
        "action": "parse", "page": PAGE, "prop": "text",
        "format": "json", "formatversion": 2})
    r.raise_for_status()
    html = r.json()["parse"]["text"]
    open(cache, "w", encoding="utf-8").write(html)
    return html


def pick_tables(html):
    tables = pd.read_html(io.StringIO(html))
    matrix = next(t for t in tables
                  if str(t.columns[0]).startswith("Mandante"))
    standings = next(t for t in tables
                     if list(t.columns)[:3] == ["Pos", "Equipevde", "Pts"])
    return matrix, standings


def build(matrix, standings):
    # mapa sigla (colunas da matriz) -> nome completo (linhas da matriz)
    codes = [c for c in matrix.columns[1:]]
    names = [str(n).strip() for n in matrix.iloc[:, 0]]
    assert len(codes) == len(names) == 20, (len(codes), len(names))
    code_of = dict(zip(names, codes))

    rows = []
    for i, home in enumerate(names):
        for code, away in zip(codes, names):
            if home == away:
                continue
            cell = str(matrix.iloc[i][code]).strip()
            m = SCORE.match(cell)
            rows.append({
                "home": home, "away": away,
                "home_goals": int(m.group(1)) if m else pd.NA,
                "away_goals": int(m.group(2)) if m else pd.NA,
                "played": bool(m),
            })
    matches = pd.DataFrame(rows)

    st = standings.rename(columns={"Equipevde": "team", "Pos": "pos", "Pts": "pts",
                                   "J": "j", "V": "v", "E": "e", "D": "d",
                                   "GP": "gp", "GC": "gc", "SG": "sg"})
    st["team"] = st["team"].str.strip()
    st = st[["pos", "team", "pts", "j", "v", "e", "d", "gp", "gc", "sg"]]
    return matches, st, code_of


def check(matches, st):
    """Confere a matriz contra a tabela de classificacao publicada."""
    p = matches[matches.played]
    agg = {}
    for _, r in p.iterrows():
        for team, gf, ga in ((r.home, r.home_goals, r.away_goals),
                             (r.away, r.away_goals, r.home_goals)):
            a = agg.setdefault(team, dict(j=0, v=0, e=0, d=0, gp=0, gc=0))
            a["j"] += 1; a["gp"] += gf; a["gc"] += ga
            a["v" if gf > ga else "e" if gf == ga else "d"] += 1
    derived = pd.DataFrame(agg).T
    derived["pts"] = derived.v * 3 + derived.e
    join = st.set_index("team").join(derived, rsuffix="_calc")
    bad = join[(join.pts != join.pts_calc) | (join.j != join.j_calc)
               | (join.gp != join.gp_calc) | (join.gc != join.gc_calc)]

    # Duas coisas MUITO diferentes cabem em "bad", e tratar as duas como fatal
    # custou caro: um editor da Wikipedia atualizou a tabela e esqueceu de
    # preencher a celula da matriz, e isso desligou a conferencia cruzada do
    # projeto inteiro por dias.
    #
    #   MATRIZ ATRASADA: a matriz tem MENOS jogos, e nada se contradiz. E uma
    #   lacuna — o que esta la continua confiavel, e o ge.globo preenche o
    #   resto. Avisa e segue.
    #
    #   CONTRADICAO: a matriz afirma algo que a tabela nega. Aborta.
    atrasada = bool(len(bad)) and bool(
        (join.j_calc <= join.j).all() and (join.gp_calc <= join.gp).all()
        and (join.gc_calc <= join.gc).all() and (join.pts_calc <= join.pts).all())
    return join, bad, atrasada


if __name__ == "__main__":
    html = fetch_html(refresh="--refresh" in sys.argv)
    matrix, standings = pick_tables(html)
    matches, st, code_of = build(matrix, standings)
    join, bad, atrasada = check(matches, st)
    print(f"jogos totais    : {len(matches)}")
    print(f"jogos disputados: {int(matches.played.sum())}")
    print(f"jogos restantes : {int((~matches.played).sum())}")
    print(f"gols/jogo       : {(matches.home_goals.sum()+matches.away_goals.sum())/matches.played.sum():.3f}")

    # A conferencia vem ANTES de gravar: se a matriz nao bate com a tabela
    # publicada, os CSVs antigos ficam intactos em vez de virarem lixo.
    if len(bad):
        cols = ["pts", "pts_calc", "j", "j_calc", "gp", "gp_calc", "gc", "gc_calc"]
        if atrasada:
            faltam = int((join.j - join.j_calc).sum() // 2)
            print(f"\nAVISO: a MATRIZ da Wikipedia esta atrasada em {faltam} jogo(s).")
            print(bad[cols].to_string())
            print("\nNada se contradiz — e lacuna, nao erro. O que a matriz tem continua")
            print("valendo, o ge.globo preenche o resto, e a conferencia cruzada segue")
            print("valida nos jogos que as duas fontes tem.")
        else:
            print("\nCONTRADICAO entre a matriz de confrontos e a tabela publicada:")
            print(bad[cols].to_string())
            print("\nA matriz afirma algo que a tabela nega — nao e so atraso.")
            print("Nada foi gravado; data/ continua com os dados da ultima rodada boa.")
            sys.exit(1)

    matches.to_csv("data/matches.csv", index=False, encoding="utf-8")
    st.to_csv("data/standings.csv", index=False, encoding="utf-8")
    json.dump(code_of, open("data/codes.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if len(bad):
        print("\ngravado, com a ressalva do atraso acima")
    else:
        print("\nconferido: matriz bate com a tabela publicada nos 20 times")
