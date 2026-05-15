# @merkie/agentic

Opinionated `streamText()`-first framework for building AI agents in JS.

The package wraps the Vercel AI SDK + OpenRouter with the bits every real agent
app keeps rewriting: session state, raw `ModelMessage[]` persistence to JSONL,
queued messages, early tool-call streaming, abort salvage, cost tracking, and
context-window math.

## Features

- **OpenRouter-first provider setup** with usage accounting enabled.
- **Session abstraction** around `streamText()` with one active run per session.
- **Queued messages** so user messages and system notifications that arrive mid-run are grouped into the next turn.
- **Abort handling** with an in-memory `AbortController` per active session.
- **Partial stream salvage** for aborted runs, including synthetic tool results for dangling tool calls so replay stays valid.
- **Raw `ModelMessage[]` persistence**: every `streamText()` input message and step output message is appended to an append-only `.jsonl` exactly as the AI SDK produced it — no custom schema, no display-string round-trip. Replay is "read the lines back in order."
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

## JSONL Persistence

Persistence is deliberately dumb: every turn appends two record kinds to a
`.jsonl` file:

- `run_start` — carries the `ModelMessage[]` that went into `streamText()` for
  that turn (the new user messages, system notifications, etc.).
- `step_messages` — carries each `ModelMessage` that came out of a step
  (assistant text, tool calls, tool results), exactly as the AI SDK emitted it.

Both record kinds store the **AI SDK's own message objects** — not display
strings, not a custom schema. Tool calls and tool results survive round trips
verbatim, which is what lets future turns continue with tool-use continuity
intact.

Replay is "open the file and read those lines back in order." There is no
reconstruction step. The same `ModelMessage[]` you'd hand to `streamText()`
comes straight out. Point a new session at an existing log file and the prior
history loads lazily on first use:

```ts
const session = agentic.getSession({
  id: "chat_123",
  logFile: "sessions/chat_123.jsonl",
});

console.log(session.history); // ModelMessage[], straight out of the JSONL

await session.send("And what did I ask before?"); // picks up where the log left off
```

You don't need to hand-feed prior messages — sending a new message reuses the
loaded history automatically. Other record kinds (`stream_event`, `run_end`,
`run_aborted`, `cost_summary`) are also appended for debugging and cost
tracking, but the message history itself comes from `run_start` + `step_messages`
alone.

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
