# @merkie/agentic

Opinionated `streamText()`-first framework for building AI agents in JS.

The package wraps the Vercel AI SDK + OpenRouter with the bits every real agent
app keeps rewriting: session state, raw JSONL replay, queued messages, early
tool-call streaming, abort salvage, cost tracking, and context-window math.

## Features

- **OpenRouter-first provider setup** with usage accounting enabled.
- **Session abstraction** around `streamText()` with one active run per session.
- **Queued messages** so user messages and system notifications that arrive mid-run are grouped into the next turn.
- **`<SystemNotification>` helper** for wakeups, background task completions, and app-level events.
- **Abort handling** with an in-memory `AbortController` per active session.
- **Partial stream salvage** for aborted runs, including synthetic tool results for dangling tool calls so replay stays valid.
- **Raw JSONL persistence**: append-only logs with `run_start`, `stream_event`, `step_messages`, `run_end`, `run_aborted`, and `cost_summary`.
- **Replay helpers** that reconstruct AI SDK `ModelMessage[]` directly from JSONL.
- **Early tool-call detection** via `tool-input-start`, `tool-input-delta`, `tool-input-end`, `tool-call`, and `tool-result` events.
- **Tool factories** for binding app/session state through closure instead of model-controlled parameters.
- **OpenRouter model catalog helpers** with TTL cache, inflight de-dupe, stale fallback, context length lookup, and fail-open model validation.
- **Cost + context tracking** from AI SDK usage and OpenRouter provider metadata.

## Install

```sh
pnpm add @merkie/agentic ai @openrouter/ai-sdk-provider zod
```

This repo itself uses `pnpm` and pins the package manager in `package.json`.

## Quickstart

```ts
import { createAgentic, createTools } from "@merkie/agentic";
import { tool } from "ai";
import { z } from "zod";

const tools = createTools(({ context }) => ({
  lookup_user: tool({
    description: "Look up the current application user.",
    inputSchema: z.object({}),
    execute: async () => ({ userId: context.userId }),
  }),
}));

const agentic = createAgentic({
  openRouterApiKey: process.env.OPENROUTER_API_KEY!,
  defaultModel: "google/gemini-3.1-flash-lite-preview",
});

const session = agentic.getSession({
  id: "chat_123",
  system: "You are a concise assistant.",
  tools,
  context: { userId: "user_123" },
});

session.onEvent((event) => {
  if (event.type === "text-delta") process.stdout.write(event.text);
  if (event.type === "tool-input-start") {
    console.log(`\ncalling ${event.toolName}`);
  }
});

await session.send("Who am I?");
```

## JSONL Replay

Every session writes a raw `.jsonl` file. The important replay invariant is:
persist AI SDK `ModelMessage[]`, not just display strings. Tool calls and tool
results must survive round trips or future turns break tool-use continuity.

```ts
import { replayJsonl } from "@merkie/agentic";

const replayed = replayJsonl("sessions/chat_123.jsonl");

// Feed this back into streamText({ messages }) or inspect it in tests.
console.log(replayed.fullMessages);
```

## System Notifications

`formatSystemNotification` wraps content in a `<SystemNotification>` block.
Send it like any other message — if a run is active it queues and is grouped
with any other pending messages on the next turn.

```ts
import { formatSystemNotification } from "@merkie/agentic";

await session.send({
  role: "user",
  content: formatSystemNotification("Background task completed.", {
    taskId: "task_123",
    status: "completed",
  }),
});
```

This becomes:

```xml
<SystemNotification>
Background task completed.

Metadata:
{ "taskId": "task_123", "status": "completed" }
</SystemNotification>
```

## Abort

```ts
const pending = session.send("Write a long report.");

setTimeout(() => {
  session.abort("User clicked stop");
}, 500);

const result = await pending;
console.log(result.aborted, result.text);
```

On abort, completed step messages and the in-progress assistant text/tool calls
are salvaged into replayable model messages. Tool calls without results receive
a synthetic interrupted tool result.

## Local Development

```sh
pnpm install
pnpm typecheck
pnpm test
pnpm build
```

Run the demo harness with a real OpenRouter key:

```sh
OPENROUTER_API_KEY=... pnpm dev:send-message -- "What's the weather in Tokyo?"
```
