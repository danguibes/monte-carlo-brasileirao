"""Fonte alternativa: API do ge.globo (nao oficial, sem chave).

Traz o que a matriz da Wikipedia nao tem: DATA e RODADA de cada jogo. E o que
liga o peso por recencia. Grava data/matches.csv com as colunas extras e
concilia contra a fonte da Wikipedia, quando ela existir.

Nao oficial quer dizer sem contrato e sem aviso de mudanca: se um dia parar de
responder, parse_data.py continua funcionando sem datas.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests

BASE = ("https://api.globoesporte.globo.com/tabela/"
        "d1a37fa4-e948-43a6-ba53-ab24ab3a45b1/fase/"
        "fase-unica-campeonato-brasileiro-2026")
UA = {"User-Agent": "Mozilla/5.0 (brasileirao-mc; uso pessoal)"}
RODADAS = 38
MIN_APOS_INICIO = 150   # 90 + intervalo + acrescimos + folga, em minutos

# Estados de transmissao vistos na fonte. So ENCERRADA e jogo acabado.
# REAL_TIME e bola rolando; LIVE aparece tambem ANTES do apito, quando a
# transmissao ja esta no ar; PRE_DIA e o dia do jogo. Qualquer um deles nunca
# pode virar "encerrado", por mais tempo que tenha passado — sem essa lista,
# um jogo parado por 2h30 (VAR, falta de luz, confusao) entraria com o placar
# parcial assim que o relogio de seguranca estourasse.
NAO_ENCERRADOS = {"LIVE", "REAL_TIME", "AO_VIVO", "PRE_DIA", "PRE_JOGO", "ADIADO"}
ENCERRADO = "ENCERRADA"
_vistos = set()


BRT = timezone(timedelta(hours=-3))


def agora_brt():
    """Agora em horario de Brasilia, seja qual for o fuso da maquina.

    datetime.now() aqui era um bug que SO aparecia no CI: o runner do GitHub
    roda em UTC, tres horas a frente, entao um jogo das 20:30 parecia ter
    comecado ha 3h e o placar parcial entrava como final. Na minha maquina,
    em BRT, o mesmo codigo acertava — que e o pior tipo de bug de fuso."""
    return datetime.now(timezone.utc).astimezone(BRT).replace(tzinfo=None)


def comecou_ha(data_realizacao, minutos):
    """O jogo comecou ha mais de `minutos`? Usado so quando a API nao diz o
    estado. A fonte publica os horarios em Brasilia."""
    if not data_realizacao:
        return False
    try:
        ini = datetime.strptime(data_realizacao[:16], "%Y-%m-%dT%H:%M")
    except ValueError:
        return False
    return agora_brt() >= ini + timedelta(minutes=minutos)

# ge.globo usa nomes curtos; a Wikipedia usa nomes longos. A ponte e explicita
# de proposito: se um nome mudar, o script para em vez de casar errado.
NOMES = {
    "Athletico-PR": "Athletico Paranaense",
    "Atlético-MG": "Atlético Mineiro",
    "Bahia": "Bahia",
    "Botafogo": "Botafogo",
    "Bragantino": "Red Bull Bragantino",
    "Chapecoense": "Chapecoense",
    "Corinthians": "Corinthians",
    "Coritiba": "Coritiba",
    "Cruzeiro": "Cruzeiro",
    "Flamengo": "Flamengo",
    "Fluminense": "Fluminense",
    "Grêmio": "Grêmio",
    "Internacional": "Internacional",
    "Mirassol": "Mirassol",
    "Palmeiras": "Palmeiras",
    "Remo": "Remo",
    "Santos": "Santos",
    "São Paulo": "São Paulo",
    "Vasco": "Vasco da Gama",
    "Vitória": "Vitória",
}


def fetch(cache="data/globo.json", refresh=False):
    if os.path.exists(cache) and not refresh:
        return json.load(open(cache, encoding="utf-8"))
    todas = []
    for n in range(1, RODADAS + 1):
        r = requests.get(f"{BASE}/rodada/{n}/jogos/", headers=UA, timeout=25)
        r.raise_for_status()
        jogos = r.json()
        for j in jogos:
            j["rodada"] = n
        todas += jogos
        print(f"  rodada {n:2d}: {len(jogos)} jogos", end="\r")
        time.sleep(0.15)
    print(f"  {RODADAS} rodadas, {len(todas)} jogos            ")
    json.dump(todas, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    return todas


def build(jogos):
    desconhecidos = set()
    rows = []
    for j in jogos:
        h = j["equipes"]["mandante"]["nome_popular"]
        a = j["equipes"]["visitante"]["nome_popular"]
        for nome in (h, a):
            if nome not in NOMES:
                desconhecidos.add(nome)
        gh, ga = j["placar_oficial_mandante"], j["placar_oficial_visitante"]
        tem_placar = gh is not None and ga is not None

        # Jogo EM ANDAMENTO tambem tem placar — o parcial. Contar isso como
        # resultado final envenena tudo: o ajuste, a tabela e a projecao.
        # A API distingue: broadcast.id e ENCERRADA, LIVE, PRE_DIA ou vazio.
        trans = j.get("transmissao") or {}
        estado = ((trans.get("broadcast") or {}).get("id") or "").upper()
        if estado == ENCERRADO:
            encerrado = tem_placar
        elif estado in NAO_ENCERRADOS:
            encerrado = False
        else:
            # Valor novo ou ausente: cai no relogio, conservador de proposito.
            # E avisa, porque estado desconhecido merece olho humano.
            if estado and estado not in _vistos:
                _vistos.add(estado)
                print(f"  [atencao] estado de transmissao desconhecido: {estado!r} "
                      "— tratado pelo relogio; confira se e jogo encerrado")
            encerrado = tem_placar and comecou_ha(j["data_realizacao"], MIN_APOS_INICIO)

        rows.append({
            "home": NOMES.get(h, h), "away": NOMES.get(a, a),
            "home_goals": gh if encerrado else None,
            "away_goals": ga if encerrado else None,
            "played": encerrado,
            "em_andamento": tem_placar and not encerrado,
            "estado": estado or None,
            "rodada": j["rodada"], "data": j["data_realizacao"],
        })
    if desconhecidos:
        sys.exit("nomes de time nao mapeados em NOMES: " + ", ".join(sorted(desconhecidos)))
    df = pd.DataFrame(rows)
    # Jogos adiados vem com data nula. Precisam ser ordenaveis, entao caem no
    # fim da fila; como nenhum deles foi disputado, nao afetam a recencia.
    # O guarda-corpo abaixo garante que essa premissa continue valendo.
    sem_data = df.data.isna() | (df.data == "")
    if (sem_data & df.played).any():
        sys.exit("ha jogo DISPUTADO sem data; o peso por recencia ficaria errado")
    df["adiado"] = sem_data
    df = (df.assign(_ord=df.data.fillna("9999"))
            .sort_values(["_ord", "home"]).drop(columns="_ord").reset_index(drop=True))
    return df


def conciliar(novo, antigo_path="data/matches.csv"):
    """Confere a fonte nova contra a da Wikipedia, jogo a jogo."""
    if not os.path.exists(antigo_path):
        return None
    old = pd.read_csv(antigo_path, encoding="utf-8")
    if "rodada" in old.columns:       # ja e desta fonte; nada a conciliar
        return None
    k = ["home", "away"]
    cols = ["home_goals", "away_goals"]
    # So o que as DUAS dao por encerrado. A Wikipedia atualiza placar ao vivo,
    # entao um jogo em andamento aparece la com parcial e aqui como nao
    # disputado — isso e o comportamento certo, nao divergencia.
    j = (old[old.played].set_index(k)[cols]
         .join(novo[novo.played].set_index(k)[cols], rsuffix="_ge", how="inner"))
    return j[(j.home_goals != j.home_goals_ge) | (j.away_goals != j.away_goals_ge)]


if __name__ == "__main__":
    print("buscando 38 rodadas no ge.globo…")
    jogos = fetch(refresh="--refresh" in sys.argv)
    df = build(jogos)
    dif = conciliar(df)

    print(f"jogos totais    : {len(df)}")
    print(f"jogos disputados: {int(df.played.sum())}")
    print(f"jogos restantes : {int((~df.played).sum())}")
    com = df.data.dropna()
    print(f"periodo         : {com.min()[:10]} a {com.max()[:10]}")
    if df.em_andamento.any():
        viv = df[df.em_andamento]
        print(f"em andamento AGORA ({len(viv)}), fora da conta:")
        for r in viv.itertuples():
            print(f"  {r.home} x {r.away}  ({r.data[11:16]}, {r.estado})")
    if df.adiado.any():
        print(f"sem data (adiados): {int(df.adiado.sum())} "
              f"na rodada {sorted(df[df.adiado].rodada.unique())}")

    if dif is not None:
        if len(dif):
            print(f"\nDIVERGENCIA contra a Wikipedia em {len(dif)} jogos:")
            print(dif.head(20).to_string())
            print("\nNada foi gravado.")
            sys.exit(1)
        print("\nconferido: ge.globo bate com a Wikipedia em todos os jogos disputados")

    df.to_csv("data/matches.csv", index=False, encoding="utf-8")
    print("gravado data/matches.csv com rodada e data")
