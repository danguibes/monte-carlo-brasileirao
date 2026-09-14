# Monte Carlo — Campeonato Brasileiro Série A 2026

**→ https://danguibes.github.io/monte-carlo-brasileirao/**

Estima a distribuição de resultado final da temporada simulando os jogos que
faltam. A página no ar se atualiza sozinha todo dia às 8h de Brasília, por uma
GitHub Action que roda `atualizar.py` e republica. Se as fontes divergirem, o
job falha e o Pages continua servindo a última versão boa.

O mesmo arquivo funciona offline: `web/brasileirao.html` abre com duplo clique,
sem servidor e sem rede.

## Fontes de dados

Duas, independentes, e é de propósito.

| fonte | o que traz | chave |
|---|---|---|
| Wikipedia pt (`parse_data.py`) | matriz de confrontos + tabela publicada | não |
| ge.globo (`fetch_globo.py`) | **data e rodada** de cada jogo, placares, estádio | não |

O ge.globo é uma API não oficial (`api.globoesporte.globo.com`) — sem contrato,
sem aviso de mudança, mas sem chave e com tudo que falta. É ela que traz a data
por jogo, sem a qual o peso por recência não existe.

**Jogo em andamento não conta.** Uma partida rolando também tem placar — o
parcial — e contá-lo como final envenena o ajuste, a tabela e a projeção. O
ge.globo distingue pelo campo `transmissao.broadcast.id`: `ENCERRADA`, `LIVE`,
`PRE_DIA` ou vazio. Só `ENCERRADA` entra; se o campo sumir ou mudar, cai num
relógio conservador de 150 minutos após o apito inicial.

O relógio de segurança usa **horário de Brasília explícito**, não o da
máquina. Com `datetime.now()` cru, o bug só aparecia no CI: o runner do GitHub
roda em UTC, três horas à frente, então um jogo das 20:30 parecia ter começado
há três horas e o placar parcial entrava como final — enquanto na máquina do
dono, em BRT, o mesmo código acertava. `testes.py` trava isso e roda como
primeiro passo de `atualizar.py`, sem tolerância a falha.

Isso **não** era pego pela conferência entre as fontes: a Wikipedia também
atualiza placar ao vivo, então as duas concordavam no parcial. O controle
cruzado prova consistência, não finalização — são coisas diferentes, e essa
foi a lição mais cara desta página. O topo do site mostra quantos jogos estão
em andamento e fora da conta, e a lista de jogos marca cada um deles.

**Matriz atrasada não é contradição.** O parser distingue dois casos que antes
tratava igual. Se a matriz tem *menos* jogos que a tabela e nada se contradiz,
é lacuna: avisa e segue, e o ge.globo preenche. Só aborta quando a matriz
**afirma algo que a tabela nega**.

A diferença custou caro para aprender: em 12/09 um editor da Wikipedia atualizou
a tabela com Grêmio 1–2 Vasco e nunca preencheu a célula da matriz. Com a regra
antiga, essa única célula esquecida **desligou a conferência cruzada do projeto
por dois dias**, e a mensagem ainda dizia "tente de novo mais tarde", como se
fosse transitório.

`atualizar.py` busca as duas e **só grava se elas concordarem jogo a jogo**.
Medido em 2026-09-12: acordo total nos 257 jogos disputados. Duas fontes
independentes batendo é o que separa "o número apareceu" de "o número está
certo".

A classificação **não é baixada** — ela é derivada dos próprios jogos. A tabela
publicada da Wikipedia fica como conferência, porque uma divergência aí
geralmente significa punição em pontos, que os jogos não mostram.

Testadas e descartadas: ESPN (`site.api.espn.com`) devolve 403,
football-data.org e api-futebol.com.br exigem chave, TheSportsDB responde mas
com cobertura pior que o ge.globo.

## Depois de cada rodada

```bash
python atualizar.py
```

Faz os passos em ordem — busca os resultados novos, reajusta o modelo e
simula, regenera a página — e **para no primeiro que falhar**. Leva ~1 min.

**O agendamento atrasa.** O cron do GitHub Actions é best-effort e enfileira
muito na hora cheia, que é quando todo mundo agenda. Medido neste repo: o cron
das 11:00 UTC disparou às **14:50** — 3h50 de atraso — e no dia seguinte não
tinha disparado até as 13:00. Por isso o horário virou minuto quebrado (11:23)
e ganhou uma segunda tentativa à tarde (15:23). Se precisar na hora, use
`publicar.py`.

O guarda-corpo é a conferência: antes de gravar qualquer coisa, o parser
reconstrói pontos, jogos e gols de cada time a partir da matriz de confrontos e
compara com a tabela de classificação publicada. Se divergir, aborta com código
de saída 1 e **não sobrescreve nada** — `data/` e `web/` ficam com a última
versão boa. O caso comum é a Wikipedia no meio de uma edição, com a tabela já
atualizada e a matriz ainda não; a solução é tentar de novo mais tarde.

Isso também significa que o dado depende de alguém ter editado a Wikipedia.
Não é instantâneo ao apito final.

### Publicar agora, sem esperar a manhã

```bash
python publicar.py
```

Dispara o mesmo workflow do agendamento, acompanha até o fim e **confere o ar,
não só o deploy** — baixa a página publicada e verifica que ela tem jogos de
verdade. Leva uns 3 minutos, quase tudo no bootstrap. `--nowait` só dispara e
sai.

Sem o terminal à mão, dá para fazer o mesmo pelo celular: aba **Actions** do
repositório → *atualizar e publicar* → botão **Run workflow**.

Se as fontes divergirem, o job falha de propósito e o site continua servindo a
última versão boa.

### Passo a passo, se precisar

```bash
python parse_data.py --refresh  # só buscar (sem --refresh usa o cache em data/page.html)
python run.py --n 50000         # só projetar
python export_web.py            # só regerar a página
python validate.py              # controle fora da amostra
```

## Como funciona

**Dados.** `parse_data.py` lê a matriz de confrontos da Wikipedia pt e produz
`data/matches.csv` (380 linhas: 257 disputadas, 123 restantes) e
`data/standings.csv`. O parser **reconcilia** a matriz contra a tabela de
classificação publicada — pontos, jogos, gols pró e contra, nos 20 times. Se
divergir, ele imprime a divergência em vez de seguir calado.

**Modelo.** Dixon-Coles (1997), em `model.py`:

```
λ_mandante  = exp(ataque[casa] + defesa[fora] + vantagem_casa)
λ_visitante = exp(ataque[fora] + defesa[casa])
```

com a correção τ para a dependência observada nos placares baixos (0-0, 1-0,
0-1, 1-1), que a Poisson independente erra. Identificação por soma(ataque)=0 e
soma(defesa)=0. Ajuste por máxima verossimilhança (L-BFGS-B) só nos jogos de
2026.

**Simulação.** Para cada jogo restante monta-se a distribuição conjunta de
placares numa grade 11×11 já com τ aplicado e renormalizada, e sorteia-se o
placar. Simular **gols** e não M/E/V é o que permite montar a tabela: o
Brasileirão desempata por vitórias, saldo e gols pró, e nenhum desses existe sem
placar. Do 5º critério em diante (confronto direto, cartões) o desempate é
sorteado.

## O que este modelo deliberadamente NÃO faz

**Não usa histórico de confronto direto.** Dois times se enfrentam 2x por ano;
o "histórico" de um par são poucos jogos espalhados por anos, com elencos e até
divisões diferentes. O sinal é dominado por ruído. O gancho existe
(`--h2h-weight` + `data/h2h.csv`, com encolhimento por número de jogos), está
desligado por padrão e **exige um arquivo histórico de verdade** — o script
aborta em vez de inventar dados.

**Não usa posição na tabela como fator.** A posição não é informação nova: ela é
*consequência* dos resultados que já entraram na estimativa de força. Usar as
duas coisas conta a mesma evidência duas vezes. E realimentar a classificação
sorteada dentro do Monte Carlo cria bola de neve — quem pega uma sequência boa
por sorte fica artificialmente mais forte e ganha mais —, o que **infla a
variância** em vez de corrigir nada. A força do time não muda porque ele subiu
duas posições num universo simulado.

Corolário: como a força é fixa, **a ordem dos jogos restantes não altera a
distribuição final**. É por isso que o modelo funciona sem saber as datas.

**Não trata vantagem de casa por time.** É um parâmetro só para a liga inteira,
estimado junto com ataque e defesa: `exp(0,365) = 1,44x` em 2026.

Parece errado — altitude, viagem, torcida e gramado são diferentes em cada
estádio. `fator_casa.py` ajusta a versão com uma vantagem por time e testa se
ela mede alguma coisa:

| | desvio entre times |
|---|---|
| observado | 0,398 log |
| num mundo onde a vantagem é igual para todos | 0,288 (p5 0,032 · p95 0,463) |

**p = 0,21 — não dá para distinguir de ruído.** Fora da amostra os 20
parâmetros extras rendem +0,0037 nats/jogo, t = 0,49.

As estimativas individuais mostram por quê: o Grêmio sai com **4,15x** de
vantagem em casa e o Santos com **0,85x** — jogaria melhor fora. São ~13 jogos
em casa por time; com amostra assim, o ajuste transforma sequência de sorte em
"caldeirão". O 4,15x do Grêmio é a cara de uma descoberta e quase certamente
não é nada.

## Limitações, medidas e não escondidas

1. **Poder preditivo por jogo é fraco.** `validate.py` divide os jogos
   disputados em treino/teste 20 vezes e compara log-loss em 1X2:

   | | log-loss |
   |---|---|
   | Dixon-Coles | 1,0789 |
   | taxa média da liga | 1,0905 |
   | uniforme (1/3) | 1,0986 |

   Ganho de +0,0116 nats/jogo sobre a taxa média da liga, t = 1,12, vencendo em
   15 de 20 repartições — **positivo mas não significante**. Isso é o esperado
   em futebol com 257 jogos de treino, e é um teto honesto: as probabilidades de
   título e rebaixamento são mais confiáveis que isso (dependem muito da tabela
   já construída, não só da vantagem por jogo), mas **não as leia como precisas
   na segunda casa decimal**.

2. **Peso por recência: disponível, e sem evidência de que ajuda.** Com as datas
   do ge.globo, `validate_tempo.py` faz o backtest que importa — treina no
   passado, prevê o futuro, janela deslizante de 20 jogos:

   | | log-loss |
   |---|---|
   | meia-vida 50 jogos | 1,0257 |
   | meia-vida 30 jogos | 1,0334 |
   | meia-vida 80 jogos | 1,0361 |
   | meia-vida 120 jogos | 1,0404 |
   | uniforme | 1,0442 |
   | taxa média da liga | 1,0739 |

   Meia-vida 50 é a melhor, com +0,0185 nats/jogo sobre uniforme — mas **t =
   0,98** com 100 previsões, ou seja, indistinguível de ruído.

   O bootstrap deu depois a régua que faltava para ver o **custo** da recência,
   que o log-loss sozinho não mostra: peso por recência joga fora amostra.
   `comparar_recencia.py` mede:

   | | amostra efetiva | erro do ataque | ruído/sinal |
   |---|---|---|---|
   | uniforme | 257 jogos | 0,164 log | 11,1% |
   | meia-vida 50 | 136 jogos | 0,218 log | **12,9%** |

   A meia-vida de 50 descarta 47% da amostra efetiva e aumenta o erro das
   forças em 33%. A razão ruído/sinal **piora**. Ou seja: o ganho de previsão,
   que já não era significante, é comprado com mais incerteza nos parâmetros —
   e as probabilidades finais mal se mexem (65,3/34,0 contra 65,7/32,6 no
   título). Padrão continua uniforme, agora por dois motivos medidos em vez de
   um. Para usar assim mesmo: `python run.py --half-life 50`.

3. **Gols/jogo do modelo (2,548) fica abaixo do observado (2,646).** Não é bug:
   a τ de Dixon-Coles torna a verossimilhança imprópria, e a grade renormalizada
   desloca um pouco a média. Efeito desprezível em pontos, pequeno em saldo.

4. **Nada sobre lesão, calendário, Libertadores, troca de técnico ou time
   reserva.** O modelo só sabe o que aconteceu em campo no Brasileirão.

## Probabilidade condicional

```bash
python run.py --n 50000 --dado "Palmeiras>Flamengo"   # mandante vence
python run.py --n 50000 --dado "Palmeiras=Flamengo"   # empate
python run.py --n 50000 --dado "Palmeiras<Flamengo"   # visitante vence
```

Pode empilhar vários `--dado`. Saída vai para `out/projecao_cond.csv`.

## Front

```bash
python export_web.py          # gera web/brasileirao.html (~95 KB)
```

Página única, autocontida, abre com duplo clique — sem servidor, sem rede. O
ajuste Dixon-Coles continua em Python (scipy); o que vai embutido na página é a
**distribuição conjunta de placares já pronta** de cada um dos 123 jogos que
faltam, numa grade 9×9 (erro máximo de truncamento em 1X2: 0,08 p.p.). O Monte
Carlo roda no navegador — 10.000 temporadas em ~0,3 s —, então fixar o resultado
de um jogo e ver a temporada inteira se reorganizar é um clique.

Tem projeção com barras, a distribuição de pontos finais de cada time com a
normal de mesma média e desvio sobreposta para comparação, e o mapa de calor
20×20 de posição final.

**Controle:** as duas implementações batem. Com os mesmos dados, o Python dá
66,89% de título para o Flamengo e o JS dá 66,9%; no condicional
`Palmeiras>Flamengo`, 50,61/49,23 contra 50,0/49,8 — dentro do erro de Monte
Carlo de 10.000 temporadas (≤ 0,98 p.p.).

## A Poisson se sustenta? Testada contra a binomial negativa

`binomial_negativa.py` responde. A Poisson impõe **variância igual à média**;
a binomial negativa relaxa isso (`Var = μ + μ²/r`) e com `r` grande vira a
Poisson de volta — então o teste é limpo.

| medida | resultado |
|---|---|
| dispersão de Pearson, **fora da amostra** | **1,018** (p5 0,82 · p95 1,24) — Poisson exige 1,00 |
| `r` estimado | 10.295, ou seja, indistinguível de infinito |
| razão de verossimilhança | −0,09 (χ² 1 gl exige 3,84) |
| AIC | Poisson 1500,8 · NB 1502,9 — a NB **perde** |
| log-loss 1X2 fora da amostra | +0,0001 nats para a NB (t = 0,56) |
| log-loss do **placar exato** | −0,0009 nats — a NB fica **pior** (t = −1,47) |

**A Poisson passa.** Quatro medidas independentes, nenhuma pede a troca.

Uma armadilha no caminho: a dispersão **in-sample** deu 0,854, sugerindo
*sub*dispersão — e isso teria virado uma conclusão errada. O μ vinha de um
modelo com 40+ parâmetros ajustado nesses mesmos jogos, então os resíduos
encolhem por construção. Refeita com μ estimado só no treino, dá 1,02. É a
mesma lição de sempre: medir com controle, não medir e pronto.

## O mercado como régua

`regua_mercado.py` mede a distância entre este modelo e as odds de fechamento,
que são o melhor preditor público que existe para futebol. **As odds não entram
no modelo** — servem só para saber quanto do erro é irredutível e quanto é meu.

Fonte: `football-data.co.uk/new/BRA.csv`, sem chave, temporadas 2012 a 2026,
com placar e odds no mesmo arquivo. Caminhada para a frente dentro de cada
temporada: o modelo só vê o que veio antes, mesma condição do mercado.

**3.440 previsões fora da amostra:**

| preditor | n | log-loss | ganha do modelo | t |
|---|---|---|---|---|
| Betfair Exchange | 575 | **0,9973** | +0,0487 | 4,22 |
| média do mercado (Shin) | 3.440 | **0,9977** | +0,0431 | 9,26 |
| média (de-vig ingênuo) | 3.440 | 0,9983 | +0,0425 | 9,25 |
| Bet365 | 302 | 1,0041 | +0,0209 | 1,21 |
| **este modelo** | 3.440 | **1,0408** | — | — |
| taxa média da liga | 3.440 | 1,0514 | −0,0106 | −1,54 |
| chutar 1/3 | 3.440 | 1,0986 | −0,0578 | −7,52 |

**O modelo andou 20% do caminho entre o baseline e o mercado.** Faltam 0,0431
nats/jogo, com t = 9,26 — não é ruído, é distância real.

E o número mais duro: mesmo com 3.440 previsões, a vantagem sobre a taxa média
da liga é de 0,0106 nats com **t = 1,54** — ainda não significante. O modelo é
melhor que chutar, mas a margem sobre o baseline mais burro que existe continua
sem passar no teste.

O de-vig de Shin ganha do ingênuo por 0,0006 nats — pouco, mas na direção certa
e consistente. Ele desloca massa para o favorito (+0,91 p.p.) e para longe da
zebra (−0,58), que é exatamente o viés que corrige.

**Armadilha encontrada no caminho:** a primeira implementação de Shin era um
no-op silencioso — passava a probabilidade já normalizada onde a fórmula pede a
bruta, a raiz sumia do intervalo, `z` caía em zero e a função devolvia o de-vig
ingênuo em 300 de 300 linhas. Rodava, não quebrava, e concluía "o método de
de-vig não faz diferença". `testes.py` agora trava isso.

## O limiar de ruído, medido em vez de suposto

A variação em p.p. só aparece quando passa do ruído da simulação. A primeira
versão do limiar estava **superestimada por dois motivos**, os dois corrigidos
depois de medir:

1. Usava `p = 0,5` em todas as células — o pior caso. Com 5 sementes em 10 mil
   temporadas, o desvio real teve **mediana 0,03 e p95 0,81 p.p.**, contra os
   1,39 que a fórmula do pior caso previa.
2. Multiplicava por `√2`, supondo duas rodadas independentes. **Não são**: a
   referência e o cenário usam a mesma semente e percorrem os jogos na mesma
   ordem, então todo jogo não fixado recebe o mesmo placar nas duas. Medido:
   para times fora do confronto fixado, a diferença dá **zero exato** em todas
   as sementes testadas.

O limiar agora é `1,96·√(p(1−p)/n)` por célula, sem o `√2`. Com 50 mil
temporadas dá 0,44 p.p. perto de 50% e 0,19 perto de 5%. Continua conservador
em relação ao ruído medido, e é de propósito.

## Linhas de corte

A página tem, no fim, a distribuição de pontos **por posição** — campeão, G4,
G6 e o 16º que escapa do Z4. É outra pergunta que a curva por time: não é
"quantos pontos o Flamengo faz", é "quantos pontos faz **quem** termina em 16º",
seja quem for.

O ponto estatístico interessante é a largura. O desvio da linha de corte é de
~1,9 ponto, contra ~4 a 5 de qualquer time individual. É uma **estatística de
ordem**: a incerteza sobre *quem* ocupa a vaga — que é enorme — desaparece, e
sobra só a incerteza sobre o nível da liga. Dá para cravar "45 salva" com
confiança sem fazer ideia de quem vai precisar dos 45.

O histograma é acumulado dentro do próprio laço da simulação (20 operações a
mais por temporada), então a seção responde aos condicionais e ao erro de
estimação como todo o resto, e vai estreitando conforme sobram menos jogos.

`linha_de_corte.py` faz a mesma conta em Python, com a tabela de "quantos
pontos bastam" que não cabia na página.

## Glossário

`web/glossario.html` explica Dixon-Coles, vantagem de casa, ρ, bootstrap,
nats/jogo, erro de Monte Carlo e a armadilha do "placar mais provável". É
**gerado** por `export_web.py` com os números da rodada, não escrito à mão —
texto fixo com número velho é pior que número nenhum, e o gerador aborta se
algum marcador ficar sem substituir.

Isso vale inclusive para o **veredito** da validação, não só para os números:
em 12/09 o ganho fora da amostra virou de +0,0116 para −0,0006 nats com quatro
jogos novos, e uma frase fixa dizendo "positivo, mas não significante" teria
passado a mentir. A frase é escolhida a partir do sinal e do t medidos.

## Saídas

- `out/projecao.csv` — pontos médios e percentis, posição mediana, P(título),
  P(G4), P(G6), P(Z4)
- `out/distribuicao_posicao.csv` — matriz 20×20, P(time terminar em cada posição)
- `out/forcas.csv` — ataque, defesa e índice por time
