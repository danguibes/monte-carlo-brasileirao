"""Invariantes que ja quebraram uma vez. Rodam antes de tudo em atualizar.py.

Nao e suite de testes: e a lista curta dos erros que passaram despercebidos e
que so aparecem em producao. Cada um aqui custou uma publicacao errada.

    python testes.py
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import fetch_globo as F


def check(nome, cond, detalhe=""):
    print(f"  {'ok  ' if cond else 'FALHA'}  {nome}" + (f"  — {detalhe}" if detalhe and not cond else ""))
    return cond


def teste_fuso():
    """O relogio tem que ser o de Brasilia, nao o da maquina.

    O runner do GitHub roda em UTC. Com datetime.now() cru, um jogo das 20:30
    parecia ter comecado ha 3 horas e o placar parcial entrava como final —
    e na maquina do dono, em BRT, o mesmo codigo acertava."""
    ok = True
    agora = F.agora_brt()
    real = datetime.now(timezone.utc).astimezone(F.BRT).replace(tzinfo=None)
    ok &= check("agora_brt independe do TZ da maquina",
                abs((agora - real).total_seconds()) < 5)

    # jogo que comecou ha 20 minutos NAO pode contar como encerrado
    recente = (agora - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M")
    ok &= check("jogo de 20 min atras nao e encerrado",
                not F.comecou_ha(recente, F.MIN_APOS_INICIO))

    antigo = (agora - timedelta(minutes=200)).strftime("%Y-%m-%dT%H:%M")
    ok &= check("jogo de 200 min atras e encerrado",
                F.comecou_ha(antigo, F.MIN_APOS_INICIO))
    return ok


def teste_estados():
    """So ENCERRADA conta. Bola rolando tem placar parcial e nao pode entrar."""
    agora = F.agora_brt()
    velho = (agora - timedelta(minutes=300)).strftime("%Y-%m-%dT%H:%M")

    def jogo(estado, gh, ga, data):
        return {"equipes": {"mandante": {"nome_popular": "Palmeiras"},
                            "visitante": {"nome_popular": "Santos"}},
                "placar_oficial_mandante": gh, "placar_oficial_visitante": ga,
                "transmissao": {"broadcast": {"id": estado}} if estado else {},
                "rodada": 1, "data_realizacao": data}

    casos = [
        ("ENCERRADA", 2, 0, velho, True, False),
        ("REAL_TIME", 1, 0, velho, False, True),    # parado ha 5h: ainda nao acabou
        ("LIVE", None, None, velho, False, False),
        ("PRE_DIA", None, None, velho, False, False),
        ("COISA_NOVA", 1, 1, velho, True, False),   # desconhecido cai no relogio
    ]
    ok = True
    for estado, gh, ga, data, esp_played, esp_viv in casos:
        df = F.build([jogo(estado, gh, ga, data)])
        r = df.iloc[0]
        ok &= check(f"{estado:<11} -> played={esp_played}, em_andamento={esp_viv}",
                    bool(r.played) == esp_played and bool(r.em_andamento) == esp_viv,
                    f"veio played={bool(r.played)}, em_andamento={bool(r.em_andamento)}")
    return ok


def teste_placar_parcial_nao_vaza():
    """Jogo nao encerrado nao pode carregar placar nenhum para o CSV."""
    agora = F.agora_brt()
    j = {"equipes": {"mandante": {"nome_popular": "Palmeiras"},
                     "visitante": {"nome_popular": "Santos"}},
         "placar_oficial_mandante": 1, "placar_oficial_visitante": 0,
         "transmissao": {"broadcast": {"id": "REAL_TIME"}},
         "rodada": 1,
         "data_realizacao": (agora - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M")}
    r = F.build([j]).iloc[0]
    import pandas as pd
    return check("placar parcial nao vaza para o CSV",
                 pd.isna(r.home_goals) and pd.isna(r.away_goals))


def teste_devig_shin():
    """Shin tem que FAZER alguma coisa. A primeira versao passava a
    probabilidade ja normalizada onde a formula pede a bruta, a raiz sumia do
    intervalo, z caia em zero e a funcao devolvia o de-vig ingenuo — em 300 de
    300 linhas. Rodava, nao quebrava, e nao media nada."""
    import numpy as np
    from regua_mercado import devig_shin, devig_simples
    o = np.array([[1.14, 7.42, 16.04], [2.10, 3.30, 3.60], [1.43, 4.35, 7.08]])
    a, b = devig_simples(o), devig_shin(o)
    ok = True
    ok &= check("Shin soma 1", np.allclose(b.sum(1), 1, atol=1e-9))
    ok &= check("Shin difere do ingenuo", np.abs(b - a).max() > 1e-3,
                f"maior desvio {np.abs(b-a).max():.6f}")
    # desloca massa para o favorito, que e o vies que ele corrige
    fav = a.argmax(1)
    dif = b[np.arange(len(o)), fav] - a[np.arange(len(o)), fav]
    ok &= check("Shin desloca para o favorito", (dif > 0).all(),
                f"deslocamentos {dif.round(4)}")
    return ok


if __name__ == "__main__":
    print("invariantes:")
    tudo = all([teste_fuso(), teste_estados(), teste_placar_parcial_nao_vaza(),
                teste_devig_shin()])
    print("\n" + ("todos passaram" if tudo else "HA FALHA — nao publique"))
    sys.exit(0 if tudo else 1)
