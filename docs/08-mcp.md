# 08 — Cliente MCP: só tools, por stdio

> Referência: [Como Claude Code funciona](https://code.claude.com/docs/pt/how-claude-code-works)

Um servidor MCP é outro processo que oferece tools. O modelo não fala com ele.
A harness fala. Nesta fase o cliente é o menor que ainda merece o nome: sobe
um processo, faz o handshake e chama tools. Não há resources, prompts,
sampling nem HTTP.

O fio é o stdin/stdout do processo. Cada mensagem é um JSON-RPC em uma linha,
sem newline no meio. O cliente manda quatro coisas, nesta ordem:

```
initialize
notifications/initialized     (aviso, sem id, sem resposta)
tools/list
tools/call                    (quando o modelo pede)
```

`initialize` diz quem é o cliente e qual versão do protocolo ele fala. A
resposta traz o que o servidor sabe fazer. `tools/list` devolve nome,
descrição e o JSON Schema dos argumentos. `tools/call` manda o nome e os
argumentos e recebe `content`, em geral um bloco `{type: "text", text: ...}`.
Se `isError` vier verdadeiro, ou se o JSON-RPC vier com `error`, isso é falha
da tool — não da harness.

## O nome e o portão

A tool entra no registry com o nome `mcp_<servidor>_<tool>`. O servidor `echo`
com a tool `echo` vira `mcp_echo_echo`. O prefixo evita colidir com `read` ou
`shell`, e os dois lados do nome têm de ser identificadores: sem espaço, sem
`/`, sem `..`.

O tipo da tool é `execute`, o mesmo de um comando. A tabela da Fase 3 não
ganha um caso novo:

| Modo | O que acontece |
| --- | --- |
| `default` | pergunta, como um `shell` |
| `plan` | nega, antes de existir `tools/call` |
| `bypass` | roda, salvo regra `deny` |

Sem terminal, "pergunta" vira negação, como já acontecia com qualquer comando.
Uma regra `deny` continua vencendo: o cliente nem é chamado, e um hook não
consegue liberar o que ela barrou. Se o portão autorizou, o PreToolUse da
Fase 7 ainda pode negar. O PostToolUse ainda pode anotar o resultado.

O `settings.json` que `config.py` já lê ganha os servidores. Não há segundo
arquivo, e o `command` é uma lista de argv — o cliente não abre um shell para
montar a linha:

```json
{
  "mcp": {
    "servers": {
      "echo": {"command": ["python", "server.py"]}
    }
  }
}
```

O nome do servidor repete nas camadas: a mais específica substitui o
`command`. Um campo `transport: "http"` é erro de settings. O CLI sobe os
servidores antes do turno e encerra o processo ao sair.

## Quando o servidor quebra

Um stdout que não é JSON, um processo que morre, um timeout: nada disso
sobe até derrubar o loop. `McpTool` transforma a falha em `ToolError`, e o
registry devolve `tool_result` com `is_error=True`. O modelo lê
`MCP server returned invalid JSON: ...` e segue, do mesmo jeito que lê um
arquivo inexistente.

O teste não usa rede. Ele sobe `tests/mcp_echo_server.py` como subprocesso
Python do próprio repositório. O servidor falso implementa os três métodos e
expõe `echo`. Um argumento `--log` grava cada `tools/call`, para o teste de
plan mode poder afirmar que a negação aconteceu antes do processo ver a
chamada. Outro teste passa `--break-json`: o `tools/call` responde `{not json`,
e o modelo recebe o erro no turno seguinte.

## O que ficou de fora

- **Evals** (Fase 9).
- Resources, prompts, sampling, autenticação e o transporte HTTP. O cliente
  não anuncia essas capacidades no `initialize`.
- O filho da Fase 6 não recebe as tools MCP. A lista dele continua sendo a
  das tools embutidas. O pai é quem chama `mcp_*`.
- Perfis de subagent e paralelismo de vários `task`.

## Perguntas para fixar

1. Por que a tool MCP se chama `mcp_echo_echo` e não `echo`?
2. O modo é `plan` e existe `allow: ["mcp_echo_echo"]`. A chamada roda?
3. O servidor escreveu `{not json` no stdout. Por que o turno do modelo
   continua?
4. O que mudaria no portão de permissão se a tool MCP fosse do tipo `read`?
5. Por que o `command` em settings é uma lista, e não uma string passada ao
   shell?
