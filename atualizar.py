"""Atualiza tudo depois de uma rodada, numa tacada:

    python atualizar.py

Busca os resultados novos nas DUAS fontes, confere uma contra a outra, reajusta
o Dixon-Coles, regrava os CSVs de projecao e regenera web/brasileirao.html.

A Wikipedia vem primeiro e serve de conferencia: o ge.globo so grava se bater
com ela jogo a jogo. Duas fontes independentes concordando e o que separa "o
numero apareceu" de "o numero esta certo".

Para em qualquer passo que falhar, e nada e sobrescrito: data/ e web/ continuam
com a ultima versao boa.
"""
import subprocess
import sys

PASSOS = [
    ("conferindo na Wikipedia", [sys.executable, "parse_data.py", "--refresh"]),
    ("buscando resultados e datas no ge.globo", [sys.executable, "fetch_globo.py", "--refresh"]),
    ("ajustando o modelo e simulando", [sys.executable, "run.py", "--n", "50000"]),
    ("medindo o erro de estimacao das forcas", [sys.executable, "bootstrap.py", "--b", "60", "--n", "20000"]),
    ("regerando a pagina", [sys.executable, "export_web.py"]),
]


def main():
    for i, (titulo, cmd) in enumerate(PASSOS, 1):
        print(f"\n=== {i}/{len(PASSOS)} {titulo} " + "=" * (44 - len(titulo)))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"\nParou no passo {i} ({titulo}). Nada depois dele foi refeito.")
            sys.exit(r.returncode)
    print("\n" + "=" * 60)
    print("pronto. abra web/brasileirao.html")


if __name__ == "__main__":
    main()
