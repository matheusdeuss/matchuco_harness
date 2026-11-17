# 06 — Subagents: uma conversa emprestada

> Referência: [Como Claude Code funciona](https://code.claude.com/docs/pt/how-claude-code-works)

Uma tarefa grande cabe mal numa conversa só. O modelo que está editando o
código não precisa das trinta leituras que serviram para achar o ponto certo:
precisa da conclusão. Um **subagent** é um segundo loop, com a própria lista
de mensagens, que devolve um texto e some. O pai continua com a janela quase
do tamanho que tinha.

A tool se chama `task`. O pai descreve o trabalho num prompt que tem de se
sustentar sozinho, porque o filho não recebe o histórico do pai.

## O que o filho é

```
pai:  [ tools ] [ system ] [ conversa do pai ... ] [ task ]
                                    │
                                    ▼
filho: [ tools ] [ system novo ] [ prompt da task ] [ read ] [ tool_result ] [ resposta ]
                                    │
                                    ▼
pai:  ... [ tool_result = texto final do filho ]
```

Três cortes, de propósito:

1. **Lista de mensagens própria.** O filho não herda o que o pai já leu. É
   isso que alivia a janela. O system prompt do filho é montado na hora em
   que a sessão dele começa — memória inclusa, se houver — e congelado para
   ele. O prompt do pai não é reescrito: continua o mesmo objeto.
2. **Profundidade 1.** O filho tem a tool `task`, senão o modelo não teria
   como tentar e o teste não teria o que recusar. A chamada volta como erro
   antes de existir um neto: "maximum depth is 1". Não se abre uma terceira
   sessão.
3. **Somente leitura, por padrão.** O filho entra em plan mode, o modo da
   Fase 3 que recusa edição e comando. `exit_plan_mode` não vai na lista
   dele, então ele não pede para sair desse modo. `readonly: false` copia o
   modo do pai em vez de forçar plan mode.

## Permissão não sobe

"Não exceder o pai" não é um slogan. `child_policy` copia as listas `allow`,
`ask` e `deny` para objetos novos (um "always" do filho não grava regra no
pai) e escolhe o modo assim:

| Pai | `readonly` | Filho |
| --- | --- | --- |
| qualquer | `true` (padrão) | `plan` |
| `bypass` | `false` | `bypass` |
| `plan` | `false` | `plan` |

Plan mode nega escrita **antes** de uma regra `allow`. Um pai em `bypass`
com `readonly` padrão continua podendo escrever; o filho, não. Uma regra
`deny` do pai vem junto, e deny ganha de tudo — inclusive de plan mode, que
nem chega a ser consultado. O filho não lê um `.env` que o pai tem proibido.

O resultado que volta é **um** `tool_result` de texto: a resposta final do
filho, não o rastreio das leituras. O pai não paga o conteúdo dos arquivos
duas vezes. Se o filho estoura `--max-steps`, o `tool_result` é erro.

## A sessão do filho

O filho ganha um JSONL próprio em `.matchuco/sessions/<id>.jsonl`, no mesmo
formato da Fase 5. O `tool_result` do pai carrega `child_session_id`. Dá para
abrir aquele arquivo e ver o `read` que o pai nunca recebeu por extenso.
Resume e fork do pai não copiam a conversa do filho para dentro da do pai; o
id é a referência.

## Como o teste dirige os dois com um FakeProvider

O provedor falso é uma fila. Pai e filho compartilham a mesma instância, então
um roteiro só cobre os dois:

```
1. pai pede task
2. filho pede read
3. filho responde em texto
4. pai responde em texto, já com o tool_result do filho
```

O segundo request que o filho faz não contém o prompt do usuário do pai. O
request seguinte do pai não contém o `tool_result` numerado do `read`: contém
a frase que o filho devolveu. Outro teste, com o pai em `bypass`, pede ao
filho um `write` e um `task` aninhado. O arquivo fica intacto, os dois
resultados voltam com `is_error`, e só existem duas sessões no disco — a do
pai e a do filho.

## O que ficou de fora

- **Hooks e skills** (Fase 7), **cliente MCP** (Fase 8) e **evals** (Fase 9).
- Subagents especializados com prompt e ferramentas próprias, no estilo dos
  perfis do Claude Code. Aqui o filho usa as tools do pai, menos
  `exit_plan_mode`, e o prompt é o que o modelo passou em `task`.
- Paralelismo de verdade: os tool calls de uma resposta ainda rodam um depois
  do outro. Dois `task` na mesma resposta não dividem o provedor ao mesmo
  tempo; com o `FakeProvider` a fila ficaria ambígua.

## Perguntas para fixar

1. Por que o filho não recebe o histórico do pai, se isso o obrigaria a reler
   arquivos que o pai já leu?
2. O que aconteceria com o system prompt do pai se a memória que o filho
   carregou fosse escrita de volta nele no meio do turno?
3. O pai está em `bypass` e o filho está em plan mode. Qual dos dois pode
   escrever, e por que isso não é "o filho excedeu o pai"?
4. Por que a recusa do segundo nível é um `tool_result` de erro, e não uma
   exceção que derruba o pai?
5. O JSONL do pai aponta o id da sessão do filho em vez de colar a conversa
   dele. O que se preserva com essa escolha?
