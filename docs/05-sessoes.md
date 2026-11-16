# 05 — Sessões: o que sobrevive quando o processo acaba

> Referência: [Como Claude Code funciona › Checkpoints e permissões](https://code.claude.com/docs/pt/how-claude-code-works#stay-safe-with-checkpoints-and-permissions)

O modelo não guarda nada entre processos. Fechar o terminal e abrir de novo,
sem uma harness que lembre, é começar do zero: o histórico, os arquivos que
ele já leu e o ponto em que um `write` deu errado somem junto com a memória
do processo. Esta fase torna a conversa durável, e dá um jeito de desfazer
um arquivo sem reescrever o passado.

Três mecanismos, com trabalhos diferentes:

1. **Sessão** — a conversa, em disco, para continuar ou ramificar.
2. **Checkpoint** — os bytes de um arquivo imediatamente antes de mudarem.
3. **Auto memory** — notas curtas que a próxima sessão deve ver no system prompt.

## A mesma regra da Fase 4, agora no disco

O histórico em memória já era append-only. O transcript segue a mesma regra,
por um motivo a mais: rewind e fork precisam do que aconteceu, não só do que
sobrou. Apagar uma linha para "desfazer" destruiria a única cópia de como se
chegou ali, e impediria um fork de divergir a partir do mesmo passado.

```
.matchuco/sessions/<id>.jsonl      uma linha = um evento, só se acrescenta
.matchuco/checkpoints/<id>.json    os bytes anteriores de um arquivo
.matchuco/memory.md                notas para a próxima sessão
```

Cada linha do JSONL é um evento tipado:

| Evento | O que registra |
| --- | --- |
| `user_message` | o prompt, com os lembretes que foram junto |
| `model_message` | a resposta do modelo, inclusive tool calls |
| `tool_result` | um resultado, e o mapa de arquivos já lidos depois dele |
| `compaction` | o resumo que substituiu o histórico (as linhas antigas ficam) |
| `checkpoint` | o id de um snapshot; os bytes ficam no diretório de checkpoints |
| `rewind` | "reconstrua ignorando o trecho até este checkpoint" |

O id do checkpoint entra no JSONL. Os bytes não: uma linha do tamanho do
arquivo deixaria o transcript ilegível e caro de copiar num fork. O arquivo
em `.matchuco/checkpoints/` guarda o conteúdo em base64, se o arquivo existia,
e o mtime. Isso basta para pôr o disco de volta.

Um turno que falha no meio — erro do provedor, interrupção — não é gravado.
O loop já restaura o histórico em memória nesse caso; gravar o turno pela
metade faria o `resume` ressuscitar uma conversa que o processo descartou.

## resume e fork

`resume(id)` lê o JSONL e reconstrói a lista de mensagens, o que ainda está
por enviar (um resumo de compactação que esperava o próximo prompt) e o mapa
`files_read`. Esse mapa é o que sustenta a regra de ler antes de escrever
depois de reabrir: o mtime guardado continua valendo, então um arquivo que
mudou em disco enquanto a sessão estava fechada ainda exige uma nova leitura.

`fork(id)` copia os bytes do arquivo para um id novo e não abre o original
para escrita. Os dois logs apontam para os mesmos checkpoints, que são
imutáveis. Continuar no fork — ou dar rewind nele — não altera uma linha do
log de origem. O workspace, porém, é um só: rewind mexe nos arquivos de
verdade, e isso aparece para as duas sessões.

O system prompt é montado de novo quando o processo começa (é o início dessa
execução) e em seguida congelado, como na Fase 4. Resume não reescreve o
prompt no meio de um turno.

```bash
uv run matchuco --resume abc123
uv run matchuco --fork abc123
```

`/clear` não apaga o arquivo antigo. Ele abre um id novo e relê instruções e
memória, porque uma conversa nova é uma sessão nova.

## Checkpoint e rewind

`write` e `edit` chamam o checkpoint só depois da validação e da permissão,
e imediatamente antes de mudar o arquivo. Uma chamada negada não cria
snapshot: nada ia mudar. Vários arquivos pedidos na mesma resposta do modelo
compartilham um `group`.

No log, os checkpoints daquela resposta são escritos **antes** da mensagem
do modelo. O snapshot em si foi tirado antes do `write`, mas a posição no
arquivo é o que o rewind usa como corte. Ignorar os eventos entre o
checkpoint e o rewind então derruba a tool call e o `tool_result` juntos. O
provedor nunca recebe um `tool_use` órfão — o mesmo invariante do loop.

```
... prompt do usuário
checkpoint a.txt          ← corte
checkpoint b.txt          ← mesmo group
mensagem do modelo        ← write a.txt, write b.txt
tool_result, tool_result
rewind → b.txt
```

Reconstruir para no checkpoint. As duas escritas saem da conversa que o
modelo vê. Os arquivos dos dois checkpoints do grupo, e de qualquer
checkpoint posterior, voltam aos bytes anteriores: a conversa esqueceu a
resposta inteira, então o disco não pode ficar pela metade. Um rewind para
um checkpoint de um turno anterior não desfaz um turno que ficou antes dele.

O evento `rewind` é acrescentado. As linhas do write continuam no arquivo,
para quem quiser ler o que foi tentado. `/rewind` sem id lista os
checkpoints ainda ativos; `/rewind <id>` restaura.

O mtime gravado no evento de rewind é lido do disco depois da restauração.
Assim a checagem "o arquivo mudou desde a leitura" compara com o que está
lá agora, e não com um número que o sistema de arquivos arredondou.

## Auto memory

`remember` anexa uma nota em `.matchuco/memory.md`. É uma tool de edição: o
portão de permissão vale para ela como vale para `edit`.

A nota **não** entra no system prompt deste turno. O prompt foi montado no
início da sessão e permanece o mesmo objeto. Reescrevê-lo invalidaria o
prefixo em cache e os blocos de raciocínio amarrados a ele — a regra de ouro
da Fase 4. A nota aparece na próxima sessão, ou depois de `/clear`, que
monta o prompt outra vez.

Memória ausente, vazia ou só com espaços não acrescenta parte nenhuma ao
prompt. O texto que o modelo vê é o mesmo da Fase 4.

## Como experimentar

```bash
uv run matchuco
# a primeira linha mostra `session <id>`
# depois de um write ou edit permitido, aparece `checkpoint <id>`
> /rewind          # lista os checkpoints ainda ativos
> /rewind <id>     # restaura os arquivos e continua; o JSONL cresce

uv run matchuco --resume <id>
uv run matchuco --fork <id>
```

O provedor fake não edita arquivo nenhum: o roteiro dele só devolve texto.
Para ver um checkpoint nascer, use um provedor de verdade ou o teste
`tests/test_sessions.py`, que roda o loop com o `FakeProvider` scriptado.

## O que ficou de fora

- **Subagents**, que abrem uma sessão filha e apontam o id dela num
  `tool_result`. É a Fase 6.
- **Hooks, skills, MCP e evals** (Fases 7–9).
- Um rewind não é um commit git. Ele restaura os arquivos que a harness
  snapshotou, não o working tree inteiro.

## Perguntas para fixar

1. Por que o rewind acrescenta uma linha em vez de apagar as linhas do write?
2. O que quebraria no próximo request se o checkpoint ficasse *depois* da
   mensagem do modelo que pediu o `write`?
3. Por que `remember` não altera o system prompt que já está em uso?
4. Um `resume` reabre a conversa e o mapa de arquivos lidos. Por que o mtime
   guardado ainda importa, se o arquivo "é o mesmo"?
5. Fork copia o log e não os bytes dos checkpoints. O que isso implica quando
   se dá rewind no fork?
