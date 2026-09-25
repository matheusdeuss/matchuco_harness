# 01 — Provedores: um formato neutro para vários modelos

## O problema

Cada API de LLM representa uma conversa de um jeito diferente. O ponto mais
delicado é o **tool use**:

| | Anthropic (Messages API) | OpenAI (Chat Completions) |
| --- | --- | --- |
| Pedido de ferramenta | bloco `tool_use` dentro do `content` do assistant | lista `tool_calls` no assistant |
| Argumentos | objeto JSON (`input`) | **string** JSON (`arguments`) |
| Resultado | bloco `tool_result` dentro de uma mensagem `user` | mensagem separada com `role: "tool"` |
| Raciocínio | blocos `thinking` com `signature`, devolvidos intactos | não fica no histórico |
| Streaming de tool call | eventos por bloco | fragmentos por `index` para remontar |

Se o loop agentic conhecesse esses detalhes, cada provedor novo exigiria mexer
no núcleo. A solução é o padrão **Adapter**.

## O desenho

```
            ┌──────────── núcleo (loop, tools, sessões) ────────────┐
            │  só conhece: Message, ContentBlock, ToolSpec, Response  │
            └───────────────────────┬────────────────────────────────┘
                                    │ Provider.stream(system, messages, tools)
            ┌───────────────────────┼─────────────────────────┐
     AnthropicProvider      OpenAICompatProvider         FakeProvider
     (SDK anthropic)        (OpenAI, Ollama, LM Studio)   (testes)
```

- [`messages.py`](../src/matchuco/messages.py): os tipos neutros. São modelos
  pydantic com o campo `type` como discriminador, então uma conversa inteira
  vira JSON e volta sem perda. As sessões da Fase 5 dependem disso.
- [`providers/base.py`](../src/matchuco/providers/base.py): o `Protocol`
  `Provider` e os eventos de streaming (`TextDelta`, `ThinkingDelta`,
  `ToolUseStart`, `Done`).

### Por que streaming com um evento `Done` no fim?

A UI quer imprimir os tokens conforme chegam. O loop agentic só precisa da
resposta completa. Um único gerador atende os dois: a UI consome os deltas e o
loop espera o `Done`, como faz a função `complete()`.

### Blocos que só um provedor entende

O `ThinkingBlock` da Anthropic precisa voltar **idêntico**, com assinatura, na
próxima requisição, e só para a Anthropic. Por isso guardamos o payload
original em `raw` e cada adapter decide: "é meu? reenvio. Não é? descarto."
O `OpaqueBlock` generaliza a mesma ideia para qualquer bloco futuro que a
harness não conhece. Assim dá para **trocar de provedor no meio da conversa**
sem quebrar o histórico.

## Detalhes por provedor

**Anthropic** ([`anthropic.py`](../src/matchuco/providers/anthropic.py))
- *Prompt caching*: o `cache_control` no topo da requisição cacheia o maior
  prefixo estável. Num loop agentic, que reenvia o histórico a cada turno,
  isso corta muito o custo de input.
- *Adaptive thinking* com `display: "summarized"`, para o CLI mostrar o raciocínio.
- *Eager input streaming*: o input das ferramentas chega em streaming. Se o
  JSON vier quebrado, vira um `ProviderError(retryable=True)`.
- *Refusal fallbacks* (`claude-opus-5`, `claude-fable-5-1`): se o modelo
  recusar por política, o servidor tenta um modelo de fallback na mesma chamada.

**OpenAI-compatível** ([`openai_compat.py`](../src/matchuco/providers/openai_compat.py))
- Usa Chat Completions, o "denominador comum" que Ollama, LM Studio e vLLM
  também falam.
- `_Accumulator` remonta as tool calls a partir dos fragmentos do stream.
- JSON inválido nos argumentos não é descartado: vira
  `{"__invalid_json__": ...}` para a camada de ferramentas devolver um erro ao
  modelo, que então pode se corrigir.

**Fake** ([`fake.py`](../src/matchuco/providers/fake.py)): roteiro fixo de
respostas, que grava cada requisição recebida para os testes conferirem
exatamente o que a harness enviou.

## Para experimentar

```bash
uv run matchuco --provider fake -p "oi"
uv run matchuco --provider openai
uv run pytest tests/test_openai_provider.py -v
```

## Perguntas para fixar

1. Por que as tool results da OpenAI precisam vir *antes* do texto do usuário
   na conversão?
2. O que aconteceria se descartássemos o `signature` de um bloco de thinking?
3. Por que o `FakeProvider` faz *deep copy* das mensagens que recebe?
