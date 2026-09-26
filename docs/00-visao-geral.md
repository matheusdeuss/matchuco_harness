# 00 — Visão geral: o que é uma agentic harness

> Referência: [Como Claude Code funciona](https://code.claude.com/docs/pt/how-claude-code-works)

Um LLM sozinho só recebe texto e devolve texto. Ele não lê arquivos nem roda
comandos. A **harness** é o programa em volta do modelo que:

1. **Monta o contexto**: system prompt, histórico, instruções do projeto
   (CLAUDE.md) e descrições das ferramentas.
2. **Executa as ferramentas** que o modelo pede (ler arquivo, rodar teste...).
3. **Devolve o resultado** ao modelo e repete até ele terminar: é o *loop agentic*.
4. **Garante segurança e durabilidade**: permissões, checkpoints, sessões.

```
prompt → [modelo decide] → tool call → [harness executa] → resultado → [modelo decide] → ... → resposta final
```

A doc do Claude Code descreve o loop em três fases que se misturam:
**reunir contexto → agir → verificar**. Quem escolhe a próxima ação é o modelo.
A harness só oferece as ferramentas e aplica as regras.

## Mapa conceito → fase deste projeto

| Conceito (doc do Claude Code) | Fase | Módulo |
| --- | --- | --- |
| Modelos (vários, trocáveis) | 1 | `providers/` |
| Loop agentic + ferramentas | 2 | `agent.py`, `tools/` |
| Modos de permissão e regras | 3 | `permissions.py`, `config.py` |
| Janela de contexto, compactação, AGENTS.md/CLAUDE.md | 4 | `context.py`, `agent.py` |
| Sessões JSONL, resume/fork, checkpoints, auto memory | 5 | `sessions/`, `checkpoints/`, `memory/` |
| Subagents | 6 | `extensions/subagents.py` |
| Hooks, skills | 7 | `extensions/` |
| MCP | 8 | `extensions/mcp_client.py` |
| (extra) Evals | 9 | `evals/` |

## Decisões de projeto

- **Python**: é a linguagem padrão das vagas de AI engineering, com SDKs de
  primeira linha e o ecossistema de evals.
- **Abstração de provedor própria** (sem LiteLLM): desenhar essa interface é a
  parte mais instrutiva, e é o que um revisor de portfólio vai querer ver.
- **Testes offline**: o `FakeProvider` roda um roteiro fixo de respostas, então
  o loop inteiro é testável sem rede e sem gastar tokens.
