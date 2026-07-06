# @merkie/agentic

> A resilience, storage, and observability harness for LLM agents on the
> [Vercel AI SDK](https://sdk.vercel.ai) + [OpenRouter](https://openrouter.ai).

Most AI SDK setups are bring-your-own-everything: every project re-implements
retry loops, chat persistence, crash recovery, compaction, and cost tracking —
slightly differently, with the same bugs. `@merkie/agentic` ships that
plumbing once, battle-tested, so your code is just models, prompts, and tools.

**What the harness guarantees:**

- **Runs don't fail when waiting a second would have saved them.** Transient
  provider errors (429s, 5xxs, severed SSE streams, stalled connections) are
  classified and retried with capped exponential backoff (server `Retry-After`
  wins). Deterministic errors — billing, auth, policy, malformed requests,
  context overflow — fail fast instead of burning credits in a retry loop.
- **Runs survive process restarts.** Every model step is persisted the moment
  it finishes; the agent loop is stateless over an append-only event ledger,
  so recovery from a SIGKILL mid-run is just "run the loop again". Bring your
  own storage (Prisma, SQLite, Redis…) by implementing two methods; JSONL
  file storage is built in.
- **Workflows have guaranteed outcomes.** `task()` gives the model
  `submit_deliverable` + `cancel_task`, validates the deliverable with zod
  *inside the tool* (validation errors go back to the model as tool results
  it can fix — no memoryless structured-output retries), and pokes the model
  if it ends its turn without calling a terminal tool. You always get
  `submitted | cancelled | failed`, never a throw.
- **Messages sent mid-run queue into the run — durably.** A `send()` while
  the agent is working is appended to the ledger *first* (a crash can never
  drop it), then folded into the live run at its next step boundary; the run
  doesn't end while unanswered input is waiting. Queued messages orphaned by
  a restart are picked up by `resume()` like any interrupted work.
- **Chats outlive the context window.** Compaction triggers on real
  provider-reported token counts against the model's actual context window
  (fetched from OpenRouter), summarizes into a hand-off message, and keeps
  going — silently between turns, or mid-run for agents deep in a task.
- **Cost is tracked correctly**, including BYOK: OpenRouter credits report
  `cost`; BYOK reports the provider charge in `upstream_inference_cost`.
  Per-step usage/cost is persisted, aggregated per run and per session.

## Install

```bash
npm install @merkie/agentic ai @openrouter/ai-sdk-provider zod
```

`ai` (v6), `@openrouter/ai-sdk-provider`, and `zod` (v4) are peer
dependencies. Reads `OPENROUTER_API_KEY` from the environment by default.

## The harness in 30 seconds

```ts
import { createAgentic, fileStorage } from "@merkie/agentic";
import { tool } from "ai";
import { z } from "zod";

const agentic = createAgentic({ storage: fileStorage("./.agentic") });

// ── durable chat ──────────────────────────────────────────────────────
const chat = agentic.session("chat:user-123", {
  model: "qwen/qwen3.7-max",
  system: "You are a helpful assistant.",
  tools: { /* your tools */ },
  compaction: { limit: 0.3 },          // compact at 30% of context window
});
const reply = await chat.send("hey!", { onPart: (p) => {/* stream to UI */} });

// after a crash/deploy, on boot:
for (const id of await agentic.interruptedSessions()) {
  // re-supply the agent config and pick up where the run left off
  await agentic.session(id, myAgentFor(id)).resume();
}

// ── workflow task with a guaranteed outcome ───────────────────────────
const outcome = await agentic.task({
  agent: {
    model: "qwen/qwen3.7-max",
    system: "You are a bank task worker.",
    tools: { get_account },
  },
  prompt: "Look up ACC-1001's balance for its authenticated owner and submit it.",
  deliverable: z.object({ accountId: z.string(), balance: z.number() }),
});
// outcome.status: "submitted" (typed deliverable) | "cancelled" (model's
// escape hatch, with reason) | "failed" (bounded retries exhausted)
```

Every session — chats, workflows, one-shots — shares the same ledger, so all
of it is resumable, auditable, and cost-tracked. Observability is one hook:

```ts
createAgentic({ onEvent: (e) => log(e) })
// run-start · step · retry · compaction · poke · queued-message · run-end
```

## À-la-carte helpers

Everything the harness is built from is exported for use with plain
`streamText`/`generateText`:

| Helper | What it does |
|---|---|
| `createOpenRouter` | drop-in provider factory with usage accounting on |
| `logStream` | pretty-print a full stream with live token/cost accounting |
| `withRetries(fn)` | retry any model call on transient failures, fail fast on deterministic ones |
| `classifyFailure(err)` | `transient` \| `context-overflow` \| `fatal` (+ Retry-After) |
| `createResilientFetch` | header + SSE-idle stall detection for hung connections |
| `sanitizeConversation` | heal interrupted/malformed tool-call transcripts before replay |
| `guardToolResultSizes` | cap tool results so one result can't blow the context window |
| `extractStepUsage` | per-step tokens + BYOK-reconciled cost from provider metadata |
| `getContextWindow` | a model's context length from OpenRouter, memoized |

## Local playground

`playground/mvp/` has runnable proof demos (bring an `OPENROUTER_API_KEY` in
`.env`):

```bash
npx tsx playground/mvp/demo-task.ts        # schema self-heal + guaranteed outcome
npx tsx playground/mvp/demo-chaos.ts       # injected 500s + severed SSE mid-run
npx tsx playground/mvp/demo-restart.ts     # SIGKILL mid-run → resume in new process
npx tsx playground/mvp/demo-compaction.ts  # memory survives two compactions
npx tsx playground/mvp/demo-queue.ts       # send() mid-run queues into the live run
npx tsx playground/mvp/before-after/before.ts  # the plumbing you'd write by hand
npx tsx playground/mvp/before-after/after.ts   # the same workflow on the harness
```

## Development

```bash
npm test          # vitest
npm run typecheck
npm run build     # tsup → dist/
```

## License

MIT © Merkie
