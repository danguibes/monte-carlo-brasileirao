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

Faz os três passos em ordem — busca os resultados novos, reajusta o modelo e
simula, regenera a página — e **para no primeiro que falhar**. Leva ~1 min.

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

**Não trata vantagem de casa por confronto.** É um parâmetro só, estimado junto
com ataque e defesa. Mediu-se `exp(0,365) = 1,44x` em 2026.

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

## Glossário

`web/glossario.html` explica Dixon-Coles, vantagem de casa, ρ, bootstrap,
nats/jogo, erro de Monte Carlo e a armadilha do "placar mais provável". É
**gerado** por `export_web.py` com os números da rodada, não escrito à mão —
texto fixo com número velho é pior que número nenhum, e o gerador aborta se
algum marcador ficar sem substituir.

## Saídas

- `out/projecao.csv` — pontos médios e percentis, posição mediana, P(título),
  P(G4), P(G6), P(Z4)
- `out/distribuicao_posicao.csv` — matriz 20×20, P(time terminar em cada posição)
- `out/forcas.csv` — ataque, defesa e índice por time
