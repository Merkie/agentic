# agentlib

> Batteries-included helpers for building LLM-powered apps with the [Vercel AI SDK](https://sdk.vercel.ai) and [OpenRouter](https://openrouter.ai).

Most AI SDK setups are bring-your-own-everything. `agentlib` is the start of a
framework that ships the boring-but-essential plumbing — observability,
persistence, and resilience — so you can focus on your agent code instead of
copy-pasting the same cost-tracking and logging boilerplate into every project.

This first release ships two things:

- **`createOpenRouter`** — a drop-in replacement for the provider factory that
  auto-enables usage/cost accounting and reads your API key from the
  environment.
- **`logStream`** — pretty-prints an AI SDK stream in real time (reasoning,
  messages, tool calls/results) with live token and cost accounting.

> **Status:** early. The logging here will eventually be backed by a real
> persistence/observability layer; `chalk` and friends are temporary.

## Install

```bash
npm install agentlib ai @openrouter/ai-sdk-provider
```

`ai` and `@openrouter/ai-sdk-provider` are peer dependencies — you bring the
versions your app already uses.

Works in TypeScript and JavaScript, ESM and CommonJS.

## Usage

```ts
import { streamText } from "ai";
import { createOpenRouter, logStream } from "agentlib";

// Reads OPENROUTER_API_KEY from the environment and turns on usage tracking.
const openrouter = createOpenRouter();

const result = streamText({
  model: openrouter("openai/gpt-4o-mini"),
  prompt: "Explain quantum tunneling in one paragraph.",
});

await logStream(result.fullStream);
```

CommonJS:

```js
const { createOpenRouter, logStream } = require("agentlib");
```

## API

### `createOpenRouter(settings?)`

Same signature and return type as `createOpenRouter` from
`@openrouter/ai-sdk-provider`, with two defaults applied:

| Default | Behavior | Override |
| --- | --- | --- |
| `apiKey` | Falls back to `process.env.OPENROUTER_API_KEY` | Pass `apiKey` |
| `extraBody.usage.include` | `true` (returns cost + token usage) | Pass `extraBody.usage` |

Every option you pass wins over the defaults, so it's 1:1 compatible with the
upstream factory:

```ts
const openrouter = createOpenRouter({
  apiKey: myKey,
  extraBody: {
    transforms: ["middle-out"],
    usage: { include: false }, // opt back out if you want
  },
});
```

### `logStream(stream)`

Consumes an `AsyncIterable` of AI SDK `TextStreamPart`s (i.e. the `fullStream`
from `streamText`) and prints each event as it arrives. When the model is served
through `createOpenRouter`, it also reports per-run cost and context-window
usage on completion.

```ts
await logStream(result.fullStream);
```

## License

MIT © Merkie
