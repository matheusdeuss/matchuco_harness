# 04 — Context engineering: o que o modelo vê

> Referência: [Como Claude Code funciona › A janela de contexto](https://code.claude.com/docs/pt/how-claude-code-works#the-context-window)

O modelo não tem memória entre requisições. Tudo o que ele "sabe" sobre a
tarefa está no que a harness manda em **cada** chamada: system prompt,
definições das ferramentas e a conversa inteira. Isso é a janela de contexto.
Ela tem um tamanho finito, e cada token enviado custa dinheiro e tempo.

*Context engineering* é decidir o que entra nessa janela, em que ordem, e o que
fazer quando ela enche. Esta fase trata de três perguntas:

1. O que o modelo precisa saber antes de começar? (instruções e ambiente)
2. Quanto da janela já está ocupado? (medição)
3. O que fazer quando enche? (compactação)

## A regra de ouro: prefixo estável, histórico append-only

Antes das peças, o princípio que molda todas elas. Uma requisição é montada
nesta ordem:

```
[ tools ] [ system prompt ] [ msg 1 ] [ msg 2 ] ... [ msg N ]  ← só cresce aqui no fim
└──────────────── prefixo: idêntico entre requisições ───────┘
```

- **Prompt caching**: os provedores cacheiam o maior prefixo que não mudou
  desde a última requisição. Num loop agentic, que reenvia tudo a cada passo,
  isso barateia muito cada chamada. Na Anthropic, uma leitura de cache custa
  cerca de 10% do preço normal. Mas **qualquer byte alterado** no meio do
  prefixo invalida tudo o que vem depois.
- **Preserved thinking**: nos modelos Claude mais novos, cada bloco de
  raciocínio fica "amarrado" ao prefixo exato que o produziu. Editar uma
  mensagem antiga, ou o system prompt, faz a API recusar esses blocos com erro
  400 em contas novas.

Consequência de design: **nada no começo da requisição muda durante a
sessão.** O system prompt é montado uma vez e congelado. O que muda (modo de
permissão, resumo de compactação) é **anexado** como texto numa mensagem nova.
Foi por isso que, já na Fase 3, o aviso de plan mode entrou no prompt do
usuário, e não no system prompt.

## 1. O system prompt em três partes

[`context.py`](../src/matchuco/context.py) monta:

| Parte | Conteúdo | Por quê |
| --- | --- | --- |
| base | como agir como agente de código | comportamento geral |
| environment | SO, shell, data, branch, `git status`, últimos commits | evita que o modelo gaste tool calls descobrindo o óbvio |
| instructions | `AGENTS.md` / `CLAUDE.md` | convenções do projeto |

Dois detalhes do bloco *environment*:

- **Ele diz qual shell a ferramenta `shell` usa de verdade.** No Windows é o
  `cmd.exe`, não o bash. Sem essa informação o modelo tenta `ls` e `grep`, e
  falha.
- **É um snapshot do início da sessão.** O texto avisa isso ao modelo. O
  `git status` muda durante o trabalho, mas o system prompt não pode mudar
  junto (regra de ouro).

### Arquivos de instrução

Carregados do mais geral para o mais específico. O posterior refina o anterior:

```
~/.matchuco/AGENTS.md                  você, em todo projeto
<raiz do repo>/AGENTS.md, CLAUDE.md    o time (commitado)
... cada diretório até o workspace
<workspace>/AGENTS.local.md            você, neste projeto (fora do git)
```

A busca vai da raiz do repositório git (a pasta com `.git`) até o workspace.
Um `AGENTS.md` acima do repositório não entra. Ler tanto `AGENTS.md`, a
convenção entre ferramentas, quanto `CLAUDE.md` faz um repositório configurado
para outro agente funcionar aqui sem mudanças. Veja o
[`AGENTS.md`](../AGENTS.md) deste próprio repositório.

Arquivos grandes são truncados em 40 mil caracteres: instruções que ninguém
consegue ler inteiras também não ajudam o modelo. Para que uma edição no
`AGENTS.md` passe a valer, rode `/clear`, que começa uma conversa nova e relê
os arquivos.

## 2. Medir sem pagar por isso

Há duas fontes de verdade, cada uma com um defeito:

| Fonte | Precisão | Quando existe |
| --- | --- | --- |
| `usage` reportado pelo provedor | exata | só **depois** da requisição |
| estimativa (~4 caracteres por token) | aproximada | a qualquer momento, de graça |

A harness combina as duas. Depois de cada resposta, ela guarda uma **linha de
base**: tokens de input reportados (incluindo cache) mais os de output, e o
tamanho do histórico naquele momento. A estimativa cobre só o que foi anexado
depois disso, geralmente os resultados das ferramentas:

```
contexto ≈ base_reportada + estimativa(mensagens novas desde então)
```

O erro fica pequeno e é **corrigido a cada turno** pelo número real. Não usamos
o `tiktoken`: ele é o tokenizer da OpenAI e erraria para Claude. Também não
chamamos a API de contagem de tokens, que custaria uma requisição extra por
passo. A estimativa só decide *quando* compactar, então precisão total não
compensa o custo.

Cada provedor declara sua `context_window`: 1M nos Claude atuais, uma tabela
por prefixo nos modelos OpenAI e 32K no Ollama. Servidores locais costumam
rodar com janela menor que a do modelo, e para isso existe
`--context-window`.

`/context` mostra a quebra por categoria, e a barra inferior do REPL mostra a
porcentagem em uso:

```
context [##......................................|.........] ~21,480 / 1,000,000 tokens (2.1%); auto-compacts at 800,000
  system: base prompt                          228
  system: environment                          218
  system: instructions: AGENTS.md (project)    344
  tool definitions                           1,283
  tool results                              17,020
  ...
```

Repare no peso das **definições de ferramentas**: são 1,3K tokens pagos em
toda requisição. Por isso o Claude Code carrega as ferramentas de MCP sob
demanda (tool search). Vamos voltar a isso na Fase 8.

## 3. Compactação

Quando o contexto estimado passa de **80% da janela**, a conversa é resumida
pelo próprio modelo, e o resumo substitui o histórico inteiro. Os 20% restantes
são folga para a resposta e para a própria requisição de resumo.

### Por que compactação "simples" (tudo vira um resumo)

A tentação é manter os últimos turnos literais e resumir só o começo
(*keep-tail*). Não funciona com preserved thinking: os blocos de raciocínio dos
turnos mantidos foram gerados com o histórico completo à frente e deixam de
valer quando esse histórico vira um resumo. A outra tentação é apagar
resultados de ferramentas antigos, e isso também edita o meio do histórico.

Por isso a harness faz **compactação simples**:

```
antes:  [u1][a1][u2(tool results)][a2]...[uN]
depois: [ user: <resumo> + (prompt pendente | "continue a tarefa") ]
```

Nada do histórico antigo é reenviado. É o formato que a documentação da
Anthropic recomenda para compactação no cliente, e os modelos são treinados
para continuar a partir dele.

### Os detalhes que fazem funcionar

- **Onde compactar**: antes de *cada* requisição, não só antes de um prompt
  novo. Um loop longo de ferramentas é justamente como a conversa estoura.
  Nunca no meio de uma rodada: um `tool_use` sem o `tool_result`
  correspondente geraria uma requisição inválida.
- **Três formas de terminar**:
  1. Havia um prompt novo ainda sem resposta: ele fica fora do resumo e volta
     intacto depois dele.
  2. Foi no meio do turno, com resultados de ferramentas pendentes: o resumo
     vem seguido de "continue a tarefa de onde parou".
  3. Foi um `/compact` entre turnos: o resumo fica guardado (*carryover*) e vai
     junto com o seu próximo prompt, sem inventar uma mensagem do assistente.
- **A requisição de resumo mantém system e tools** e anexa o pedido de resumo
  no fim. Assim a conversa inteira continua sendo cache hit. O prompt termina
  com "não chame ferramentas", porque elas continuam disponíveis.
- **O prompt de resumo diz o que preservar**: pedidos e restrições do usuário
  quase literais, o estado atual, os arquivos alterados, o que foi tentado e
  descartado, os pendentes e os detalhes difíceis de reconstruir. A seção
  `## Compact Instructions` do `AGENTS.md` e o foco do `/compact <foco>` são
  anexados a esse prompt.
- **Depois do resumo**:
  - `files_read` é limpo, então o modelo precisa reler um arquivo antes de
    editar. O conteúdo que ele tinha visto saiu do contexto.
  - O plan mode é anunciado de novo, se estiver ativo.
- **Proteção contra thrashing**: se o contexto continua acima do limite logo
  depois de compactar (um arquivo gigante, por exemplo), a harness para com
  `ContextOverflowError` em vez de compactar em loop. O mesmo vale para mais de
  duas compactações no mesmo turno.
- **Falha no meio do turno**: a conversa é restaurada a partir de um snapshot
  tirado antes do prompt. Isso também desfaz uma compactação que tenha
  acontecido durante o turno que falhou.

## Como experimentar

```bash
uv run matchuco --provider openai --context-window 20000
> leia todos os arquivos de src/ e me explique a arquitetura
> /context
> /compact foque nas decisões de design
```

Com uma janela de 20K a compactação dispara rápido. Observe a barra inferior e
os avisos `compacting...`.

## O que ficou de fora

- **Context editing e compactação no servidor**: a Anthropic oferece betas que
  limpam resultados antigos ou resumem do lado do servidor, e isso não conta
  como edição do histórico. São específicos de um provedor. Aqui optamos pelo
  mecanismo que funciona em todos.
- **Imports `@arquivo` no AGENTS.md**, que o Claude Code suporta.
- **Skills e subagents**, as outras duas ferramentas de contexto da doc, vêm
  nas Fases 6 e 7.

## Perguntas para fixar

1. Por que o `git status` do system prompt não é atualizado a cada turno, se
   ele fica desatualizado?
2. Numa conversa longa, que parte da requisição o prompt caching deixa barata
   e que parte continua cara?
3. O que daria errado se a compactação guardasse os 3 últimos turnos literais
   com seus blocos de thinking?
4. Por que limpar `files_read` depois de compactar, se os arquivos não mudaram?
5. A estimativa de 4 caracteres por token erra bastante para código. Por que
   isso quase não importa aqui?
