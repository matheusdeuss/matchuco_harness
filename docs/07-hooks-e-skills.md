# 07 — Hooks e skills: o que roda em volta da tool, e o que entra no prompt

> Referência: [Como Claude Code funciona](https://code.claude.com/docs/pt/how-claude-code-works)

A Fase 3 decide se uma tool call **pode** rodar. Isso não esgota o que a
harness quer fazer no meio do caminho. Às vezes a chamada já está autorizada
e ainda assim deve ser recusada — um projeto que não aceita `write` com um
conteúdo específico, um teste que barra um comando. Às vezes a chamada rodou
e o modelo precisa de um lembrete colado no resultado, sem que isso reescreva
o system prompt. E às vezes o modelo precisa de um texto longo de instrução,
mas esse texto não pode ocupar a janela desde o primeiro request.

São dois mecanismos, os dois em `extensions/`.

## Hooks: depois do portão, nunca no lugar dele

```
modelo pede tool_use
        │
        ▼
valida os argumentos
        │
        ▼
portão de permissão          deny ──► tool_result de erro
        │ allow                         (o hook nem é chamado)
        ▼
PreToolUse                   deny ──► "Hook denied: ..."
        │ allow                         (a tool não roda)
        ▼
tool.run
        │
        ▼
PostToolUse                  pode acrescentar uma nota
        │
        ▼
tool_result no histórico
```

A ordem é a regra. Um hook não é consultado se a permissão negou, e não tem
como devolver "allow" para o que o deny já barrou: ele só é chamado quando o
portão já autorizou. Se o hook recusa, a recusa volta como `tool_result` com
`is_error=True`, no mesmo espírito da permissão. O modelo lê o motivo e muda
de rumo. O loop não vê exceção.

Um hook que levanta exceção também não derruba o turno. No PreToolUse isso
vira negação (`Hook denied: PreToolUse hook failed ...`) e a tool não roda.
No PostToolUse vira uma nota no resultado que a tool já produziu.

O PostToolUse vê o resultado — sucesso ou erro da própria tool — e pode
acrescentar texto no fim. A nota entra no `tool_result` novo, que é um
append no histórico. O system prompt congelado na Fase 4 continua o mesmo
objeto. Reescrevê-lo aqui jogaria fora o prefixo cacheado e invalidaria
blocos de raciocínio presos àquele prefixo.

O filho da Fase 6 recebe o mesmo registro de hooks do pai. Um PreToolUse que
barra `shell` no pai também barra o `shell` do filho, mesmo com o pai em
`bypass` e o filho com `readonly: false`. A permissão do filho continua sendo
a cópia da Fase 6; o hook é uma camada a mais, não uma permissão maior.

### Callable no processo, settings no mesmo arquivo

No teste o hook é uma função Python, síncrona ou async, registrada em código.
Ela recebe o nome da tool e os argumentos (e, no PostToolUse, o texto do
resultado). Devolver uma string no PreToolUse nega; devolver `None` deixa
passar. No PostToolUse a string é a nota.

O `settings.json` que `config.py` já lê ganha uma seção `hooks`. Não há
segundo sistema de config, e não há shell no meio:

```json
{
  "hooks": {
    "PreToolUse": [{"matcher": "write", "deny": "settings blocked the write"}],
    "PostToolUse": [{"matcher": "read", "note": "file contents may be stale"}]
  }
}
```

`matcher` é o nome da tool, ou um glob (`*` casa com todas). As três camadas
de settings se somam, como as regras de permissão. Cada item vira um callable
no mesmo `HookRegistry` que o teste preenche à mão. Um campo `command` é
rejeitado: hook de shell fica de fora de propósito.

A primeira negação ganha. Os hooks seguintes de PreToolUse não rodam.

## Skills: nome e descrição agora, corpo quando pedirem

Um skill é um Markdown em `.matchuco/skills/<nome>.md`:

```markdown
# Layout

Check the repository layout before editing.

O corpo. Instruções longas, exemplos, o que o modelo só precisa
se for usar este skill.
```

O título é o primeiro heading. A descrição é a próxima linha não vazia. O
resto é o corpo. O nome é o stem do arquivo (`layout.md` → `layout`).

A tool `skill` fica sempre registrada, pelo mesmo motivo de `exit_plan_mode`:
tirá-la quando não há arquivo mudaria a lista de tools, que é o começo de
toda requisição. A descrição da tool lista só `nome: descrição`. O corpo não
entra no system prompt, nem no schema, nem na primeira mensagem.

Quando o modelo chama `skill(name="layout")`, o `tool_result` é o corpo (com
o título na frente). O request seguinte contém esse texto porque o histórico
é append-only — não porque alguém editou o prompt congelado. Nome
desconhecido, caminho com `..`, ou skill sem corpo: erro de tool, e o loop
segue.

`skill` é do tipo `read`. Plan mode pode carregá-la. Uma regra
`deny: ["skill(layout)"]` continua vencendo, e o hook não é chamado.

## Como o teste dirige isso com o FakeProvider

Hooks: o roteiro pede `write` num arquivo novo, com o modo em `bypass` (senão
a permissão negaria antes e o hook nem seria a história). O callable recusa.
O arquivo não aparece, o `tool_result` é `Hook denied: ...`, e a resposta
seguinte do modelo existe — o loop não morreu. Outro teste coloca uma regra
`deny` num `read`: a lista de chamadas do hook fica vazia, e o texto do
arquivo não volta. O PostToolUse acrescenta uma frase sentinela; ela está no
request seguinte, e o objeto do system prompt é o mesmo de antes.

Skills: o primeiro request que o `FakeProvider` gravou não contém a frase do
corpo. A spec da tool `skill` contém o nome e a descrição. Depois da chamada,
o request seguinte tem o corpo dentro do `tool_result`.

## O que ficou de fora

- **Cliente MCP** (Fase 8) e **evals** (Fase 9).
- Hooks de comando, no estilo do Claude Code, que recebem JSON pelo stdin de
  um shell. Aqui o hook é uma função no processo. Settings só declara a
  negação ou a nota; lógica que olha o argumento é o callable.
- Skills em `~/.matchuco/skills/`. O catálogo é o do workspace.
- Perfis de subagent e paralelismo de vários `task`, como na Fase 6.

## Perguntas para fixar

1. Por que o PreToolUse roda depois do portão, e não antes?
2. O que aconteceria com o prompt cache se o PostToolUse reescrevesse o
   system prompt para "anotar" o que a tool devolveu?
3. Uma regra `deny` bloqueou o `read`. O hook devolve `None`, que significa
   deixar passar. A tool roda?
4. Por que o corpo do skill não pode estar na descrição da tool desde o
   primeiro request, se o modelo precisa dele para seguir a instrução?
5. A tool `skill` existe mesmo quando `.matchuco/skills/` está vazio. O que
   se preserva com isso?
