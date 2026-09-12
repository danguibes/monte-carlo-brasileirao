"""Força uma atualização agora, sem esperar as 8h da manhã.

    python publicar.py           # dispara, acompanha e confere o ar
    python publicar.py --nowait  # só dispara e sai

Roda o mesmo workflow do agendamento: busca as duas fontes, confere uma contra
a outra, reajusta o modelo, refaz o bootstrap e republica no GitHub Pages.
Leva uns 3 minutos, quase tudo no bootstrap.

Precisa do gh autenticado (`gh auth status`). Se as fontes divergirem, o job
falha de propósito e o site continua servindo a última versão boa.
"""
import json
import re
import subprocess
import sys
import time
import urllib.request

REPO = "danguibes/monte-carlo-brasileirao"
WF = "atualizar e publicar"
URL = "https://danguibes.github.io/monte-carlo-brasileirao/"


def gh(*args, check=True):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"gh falhou: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def main():
    esperar = "--nowait" not in sys.argv

    if subprocess.run(["gh", "auth", "status"], capture_output=True).returncode != 0:
        sys.exit("gh nao esta autenticado. Rode: gh auth login")

    antes = gh("run", "list", "--repo", REPO, "--limit", "1",
               "--json", "databaseId", "--jq", ".[0].databaseId") or "0"
    gh("workflow", "run", WF, "--repo", REPO)
    print("disparado. esperando o GitHub registrar o job…")

    rid = antes
    for _ in range(30):
        time.sleep(3)
        rid = gh("run", "list", "--repo", REPO, "--limit", "1",
                 "--json", "databaseId", "--jq", ".[0].databaseId") or antes
        if rid != antes:
            break
    else:
        sys.exit("o job nao apareceu; veja em "
                 f"https://github.com/{REPO}/actions")

    print(f"job {rid}: https://github.com/{REPO}/actions/runs/{rid}")
    if not esperar:
        return

    r = subprocess.run(["gh", "run", "watch", rid, "--repo", REPO,
                        "--exit-status", "--interval", "10"])
    if r.returncode != 0:
        print("\nO job falhou. O site continua no ar com a versao anterior.")
        print("A causa mais comum e divergencia entre a Wikipedia e o ge.globo")
        print("no meio de uma edicao. Veja o log e tente de novo mais tarde.")
        sys.exit(1)

    # conferir o ar, nao so o deploy
    for tentativa in range(6):
        time.sleep(8)
        try:
            with urllib.request.urlopen(URL, timeout=25) as resp:
                html = resp.read().decode("utf-8")
        except Exception:
            continue
        m = re.search(r'"played":(\d+)', html)
        n = re.search(r'"fixtures":\[', html)
        if m and n:
            print(f"\nno ar: {len(html)//1024} KB, {m.group(1)} jogos disputados")
            print(URL)
            return
    print("\ndeploy ok, mas o CDN ainda serve a versao antiga. "
          "Costuma levar mais um minuto.")


if __name__ == "__main__":
    main()
