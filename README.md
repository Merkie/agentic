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
- **Runs survive process restarts — automatically.** Every model step is
  persisted the moment it finishes; the agent loop is stateless over an
  append-only event ledger, so recovery from a SIGKILL mid-run is just "run
  the loop again". With `autoResume` configured, the harness sweeps storage
  on boot and re-drives interrupted work by itself, with four resumed runs in
  flight by default and a ledger-counted attempt cap so a run that crashes the
  process on resume can't wedge it into a restart loop. Bring your own storage
  (Prisma, SQLite, Redis…) by implementing two methods; JSONL file storage is
  built in.
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
  a restart are picked up by `resume()` like any interrupted work. Concurrent
  callers whose inputs are handled in one model pass all receive that pass's
  real terminal status, including a later failure or cancellation.
- **Chats outlive the context window.** Compaction triggers on real
  provider-reported token counts against the model's actual context window
  (fetched from OpenRouter), summarizes into a hand-off message, and keeps
  going — silently between turns, or mid-run for agents deep in a task.
- **Cost is tracked correctly**, including BYOK: OpenRouter's `is_byok` flag
  picks the billing rule. BYOK requests bill fee (`cost`) + provider charge
  (`upstream_inference_cost`); credits requests bill `cost` alone — OpenRouter
  mirrors the upstream figure on credits requests too, so summing blindly
  would double-count. When `is_byok` is absent the harness never sums.
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

const agentic = createAgentic({
  storage: fileStorage("./.agentic"),
  // crash/deploy recovery, no boot code: the harness sweeps storage for
  // interrupted work and resumes it, using this to map ids back to configs
  autoResume: (sessionId) => myAgentFor(sessionId),
});

// ── durable chat ──────────────────────────────────────────────────────
const chat = agentic.session("chat:user-123", {
  model: "qwen/qwen3.7-max",
  system: "You are a helpful assistant.",
  tools: { /* your tools */ },
  compaction: { limit: 0.3 },          // compact at 30% of context window
});
const reply = await chat.send("hey!", { onPart: (p) => {/* stream to UI */} });

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
// escape hatch or an explicit abort, with reason) | "failed" (bounded retries exhausted)
```

Every session — chats, workflows, one-shots — shares the same ledger, so all
of it is resumable, auditable, and cost-tracked. Observability is one flag —
or one hook:

```ts
createAgentic({ logs: true })            // colored console line per event
createAgentic({ onEvent: (e) => log(e) })  // or ship them anywhere
// run-start · step · retry · compaction · poke · queued-message ·
// auto-resume · run-end
```

Create one `Agentic` instance per process for a given storage provider, then
differentiate workloads with session-id namespaces and per-session agent
configs. Locks, live-run mailboxes, and recovery state are instance-local, so
two instances writing the same storage can start competing runs for one
session.

For recovery tuning, pass `autoResume: { agentFor, maxAttempts,
maxConcurrent, staggerMs }`. `maxConcurrent` defaults to `4`; `staggerMs`
still spaces the start of each resumed run, while the concurrency cap prevents
a large interrupted backlog from creating an unbounded provider stampede.

The built-in storage providers preserve binary image/audio parts. Custom
storage adapters should persist events with `encodeEvent(event)` and restore
them with `decodeEvent(json)` instead of plain JSON serialization. Framework
poke reminders are stored as `user-message` events with `meta.poke`; filter
those when projecting a user-visible transcript. Compaction summarizes older
multimodal turns as text, so binary parts only remain verbatim when they fall
inside `keepRecent`.

Queued-input causality is recorded on each `step`; raw append order alone is
not a transcript (an in-flight step can be persisted after a newly arrived
user message without having seen it). Use `session.messages()` or
`replaySession()` for projections. A durable non-error step, including a tool
call/result step, counts as having handled the queued input it records. If the
run later fails, callers receive that failure and the input is not replayed
automatically, avoiding duplicate tool side effects. An input that only saw an
empty/error step remains pending for recovery.

`onPart` is a best-effort live transport, not a durable subscription. A caller
that reconnects—or whose queued input is picked up by a different serialized
run—should use the resolved `RunResult` and `session.messages()` as the source
of truth; historical stream parts are not replayed from storage.

Aborting a send's `abortSignal` is a durable, terminal cancellation. Agentic
saves the current partial assistant step before `run-end`, returns its text in
the cancelled `RunResult`, and includes it in `session.messages()`. Completed
tool calls are paired with their real results when available, or with a
synthetic interruption result, so the saved conversation remains safe to
replay. A restart between the partial-step append and `run-end` reconciles the
run as cancelled; it is never auto-resumed as unfinished work.

Servers can also stop any run owned by the current Agentic runtime, including
background auto-resume work, without retaining the caller's `AbortController`:

```ts
if (agentic.isRunning(sessionId)) {
  agentic.cancel(sessionId, new Error("Stopped by user"));
}
```

`cancel()` returns `true` only when it newly requests cancellation. This is an
explicit whole-run stop; unlike an initiating request's disconnect signal, it
is not suppressed when other callers have durably queued input into the run.
It closes only the active run: queued inputs that run never causally saw stay
pending and automatically start one fresh successor run. Multiple pending
inputs are batched into that successor's first model pass. Once that successor
is running, a later `cancel()` targets it, so repeated Stop actions remain
deterministic.

When an automatic recovery reaches `maxAttempts`, queued-only work remains in
the ledger and can still be retried with manual `resume()`; the cap prevents a
boot loop rather than deleting accepted input.

## À-la-carte helpers

Everything the harness is built from is exported for use with plain
`streamText`/`generateText`:

| Helper | What it does |
|---|---|
| `createOpenRouter` | drop-in provider factory with usage accounting on |
| `logEvents` | the default console logger (`logs: true`), usable as an `onEvent` directly |
| `withRetries(fn)` | retry any model call on transient failures, fail fast on deterministic ones |
| `classifyFailure(err)` | `transient` \| `context-overflow` \| `fatal` (+ Retry-After) |
| `createResilientFetch` | header + SSE-idle stall detection for hung connections |
| `sanitizeConversation` | heal interrupted/malformed tool-call transcripts before replay |
| `guardToolResultSizes` | cap tool results so one result can't blow the context window |
| `extractStepUsage` | per-step tokens + `is_byok`-aware billed cost (BYOK vs credits) |
| `getContextWindow` | a model's context length from OpenRouter, memoized |
| `encodeEvent` / `decodeEvent` | binary-safe StoredEvent serialization for custom adapters |
| `serializedStorage` | serialize custom-adapter operations per session |

## Local playground

`playground/mvp/` has runnable proof demos (bring an `OPENROUTER_API_KEY` in
`.env`):

```bash
npx tsx playground/boilerplate.ts          # the smallest useful setup — copy me
npx tsx playground/mvp/demo-task.ts        # schema self-heal + guaranteed outcome
npx tsx playground/mvp/demo-media-task.ts <image> # validated multimodal task
npx tsx playground/mvp/demo-chaos.ts       # injected 500s + severed SSE mid-run
npx tsx playground/mvp/demo-restart.ts     # SIGKILL mid-run → autoResume in new process
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
