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
const reply = await chat.send("hey!", {
  onProgress: (e) => {/* forward to the UI verbatim, keyed on e.responseId */},
});

// ── workflow task with a guaranteed outcome ───────────────────────────
const outcome = await agentic.task({
  agent: {
    model: "qwen/qwen3.7-max",
    system: "You are a bank task worker.",
    tools: { get_account },
  },
  prompt: "Look up ACC-1001's balance for its authenticated owner and submit it.",
  deliverable: z.object({ accountId: z.string(), balance: z.number() }),
  // durable: false, // opt out of the ledger for a disposable one-shot
});
// outcome.status: "submitted" (typed deliverable) | "cancelled" (model's
// escape hatch or an explicit abort, with reason) | "failed" (bounded retries exhausted)
```

Sessions and tasks share the same ledger by default, so they are resumable,
auditable, and cost-tracked. Presentation helpers and other disposable work can
set `task({ durable: false })`; the task still gets retries, validation, and a
guaranteed outcome, but its generated session stays in memory. Observability is
one flag—or one hook:

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

By default the sweep resumes an interrupted run in place. Agents whose tool
state lives in process memory (a sandbox, a headless browser) should restart
instead — the ledger survives a crash but that state doesn't, and resuming
would have the model act on phantom state:

```ts
createAgentic({
  autoResume: {
    agentFor: (id) => buildAgent(id), // rebuilds tools + a fresh sandbox
    onInterrupted: () => "restart",   // discard the dead run's partial work,
  },                                  // re-run its instruction from scratch
});
```

The dead run stays in history as a failed response; its user message is
answered by the fresh run. `"fail"` closes the run with no re-drive; a
throwing hook falls back to `"resume"` (the default). Restarts count toward
the same `maxAttempts` cap as resumes.

## Reading a chat

`agentic.transcript(sessionId)` is THE read API for rendering a conversation
— config-free, because reading never runs the model (no tools or prompts to
build in a GET route). `wireTranscript` drops the raw model
`message`/`messages` payloads from each item, leaving the JSON-safe shape a
client renders:

```ts
app.get("/chats/:id/messages", async (req, res) => {
  res.json(wireTranscript(await agentic.transcript(req.params.id)));
});
```

It returns user turns and response segments in causal display order — array
order IS the display order; never re-sort by timestamp. Framework internals
(poke reminders, compaction summaries) are already filtered out; canonical
text is pre-extracted onto each item (`item.text`); every id is durable,
minted once at append time and stable across restarts, compaction, and from
the first streamed token to the terminal state; and every response item
carries its lifecycle: `streaming | completed | cancelled | failed |
interrupted`. `session.transcript()` returns the identical projection when
you already hold a session; `projectSession(events)` is the pure form for a
ledger you already hold; `projectRun(events, runId)` scopes to one run.
`session.messages()` is the model's replay view — feed it to models, don't
render it. v0.9 reads ledgers written by v0.7 or later; older data fails to
load with a descriptive error.

## Streaming with stable identities

Every id a UI needs exists before the first token arrives. `onAccepted` fires
once per send — after the user message is durably appended, before its first
progress event — and every progress event carries the `responseId` of the
transcript response item the pass streams into. Key the optimistic bubble on
it; the durable transcript confirms it, never re-keys it:

```ts
const result = await chat.send(text, {
  onAccepted: ({ messageId }) => ui.userBubble(messageId, text),
  onProgress: (e) => {
    if (e.type === "text") ui.append(e.responseId, e.delta);
  },
});
// result.messageId — this send's durable user-message id (onAccepted's)
// result.runId    — the run that causally answered it
```

No client-side id minting and no placeholder rows: while a run is in flight
the transcript already contains its `streaming` response item. Progress
events are live-only and never replayed; the transcript is the durable
source of truth. (Raw AI SDK stream parts remain available to the initiating
caller through `onPart`, for the rare app that needs them.)

## Progress over the wire

The harness emits a small JSON-serializable activity vocabulary directly —
`ProgressEvent`: `response-start` · `text` (delta + offset) · `reasoning` ·
`tool-start` (with the parsed tool input) · `tool-end` · `retry` ·
`run-end`. Forward it verbatim; neither side of the wire switches over AI
SDK part types:

```ts
await chat.send(text, {
  onProgress: (e) => res.write(`data: ${JSON.stringify(e)}\n\n`),
});
```

`retry` and `run-end` are live lifecycle markers (render "retrying…", close
the stream); the durable record of terminal state is still
`RunResult`/the transcript.

## Reconnecting

`session.attach(listener)` delivers the same `ProgressEvent` stream for
every run this process executes for a session — the currently live one and
any that start later. A reconnecting client attaches first, snapshots, then
dedupes by character offset:

```ts
const detach = session.attach((e) => res.write(`data: ${JSON.stringify(e)}\n\n`));
const snapshot = await session.transcript(); // streaming item carries partialText
// drop text events whose e.offset < that item's partialText.length;
// on every response-start, reset the expected offset to 0 and (re)key
// the bubble by e.responseId — it marks a new model pass
```

The snapshot and the stream can never double-render a character.

## Custom storage: the provider contract

A provider is `append` + `load` (+ optional `listSessions`) and must be a
dumb pipe:

- `append()` resolving means the event is durable, atomically — the framework
  acknowledges callers and advances the run on that basis.
- `load()` returns exactly what was appended, in append order, including your
  own just-resolved appends (read-your-writes).
- Events round-trip verbatim: persist with `encodeEvent` and restore with
  `decodeEvent`, not `JSON.stringify` — the codec preserves binary
  image/audio parts and carries the event schema version.
- One writer per session per process group: locks and live-run mailboxes are
  instance-local, so cross-process concurrent runs on one session are not
  supported.
- No projection, dual writes, or status columns inside `append()` — the read
  side is `transcript()`, over the events alone.

The built-in providers honor all of this (`fileStorage` durability is bounded
by the OS page cache — it does not fsync per event, so a power loss can drop
the newest events; a process crash cannot). Compaction summarizes older
multimodal turns as text, so binary parts only remain verbatim while they
fall inside `keepRecent`.

## The consumer boundary

Four things an app never does, and where each need is served instead:

| Never | Instead |
|---|---|
| Insert placeholder/optimistic rows for in-flight turns | `transcript()` projects the streaming state |
| Read or write ledger events outside your provider | `agentic.transcript()` / `projectSession` |
| Mint message/turn ids | `onAccepted` · `RunResult.messageId` · `ProgressEvent.responseId` |
| Parse raw events or stream parts into app vocabulary | `onProgress`/`attach` ProgressEvents · transcript statuses |
| Hand-strip model payloads for the client | `wireTranscript()` |

## Cancellation

Aborting a send's `abortSignal` is a durable, terminal cancellation. Agentic
saves the current partial assistant step before `run-end`, returns its text in
the cancelled `RunResult`, and keeps it in the transcript. Completed
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
