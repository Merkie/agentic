# @merkie/agentic

A resilience, storage, and observability harness for LLM agents on the Vercel
AI SDK (v6) + OpenRouter. Published to npm as `@merkie/agentic`. No runtime
framework — plain TypeScript, ESM-first, built with tsup to `dist/`.

## Commands

```bash
npm test               # vitest run (src/*.test.ts)
npm run typecheck      # tsc --noEmit
npm run lint           # biome check .
npm run lint:fix       # biome check --write .
npm run build          # tsup → dist/ (runs automatically on publish)
```

All four of test/typecheck/lint/build must pass before committing.
Formatting is Biome (tabs, 100-col width) — run `npm run lint:fix` after
editing rather than hand-formatting.

## Architecture

The core design: **the agent loop is stateless over an append-only event
ledger**. Every model step is persisted the moment it finishes
(`user-message` / `run-start` / `step` / `compaction` / `run-end` events);
any process can rebuild the conversation by replaying the ledger. Crash
recovery, resume, message queueing, and auditing all fall out of this.

- `src/agentic.ts` — public API: `createAgentic()` → `session()` (durable
  chat: `send`/`resume`/`attach`/`messages`/`transcript`/`stats`/
  `isInterrupted`), `task()` (guaranteed-outcome workflows via
  `submit_deliverable`/`cancel_task` tools), `transcript(sessionId)` (the
  config-free read: same projection + live overlay as `Session.transcript`,
  shared through one internal `transcriptOf` — reading never runs the model,
  so no AgentConfig is involved), `withRetries()`, `resumeInterrupted()`
  (the auto-resume sweep). Holds per-session locks and the live-run map.
  `Agentic.storage` was removed in v0.8: the app constructed the provider
  and holds its own reference; re-exposing it invited ledger reads/writes
  outside the projection APIs.
- `src/run.ts` — `runLoop()`, the heart: replay ledger → streamText → persist
  steps → repeat. Owns retries (via `classifyFailure`), mid-run compaction,
  poking, and the `RunMailbox` for live message queueing. Internal: driving
  the loop directly bypasses locks and live-run registration, so none of it
  is exported from the package (tests import the module directly).
- `src/replay.ts` — `replaySession(events)`: ledger → messages + usage
  totals + `interruptedRunId` + `pendingMessages` — the MODEL's view.
  Exported under the index's advanced/debug section; app rendering goes
  through projection instead.
- `src/projection.ts` — the read side: `projectRun` (causal response
  segments for one run), `projectSession` (the full renderable transcript
  behind `agentic.transcript()`/`session.transcript()`), `wireTranscript`
  (transcript minus each item's raw model `message`/`messages` — the
  JSON-safe client payload; dumb by design, no options), `textOf`,
  `StreamContext`. `ProjectedInput` is public as `{ id, at, message, meta?,
  queued }`; ledger position/queue ids/settlement are projection-internal.
- `src/progress.ts` — the `ProgressEvent` vocabulary the harness emits
  (`response-start`/`text`/`reasoning`/`tool-start` (with parsed
  input)/`tool-end`/`retry`/`run-end`) plus `partProgress`, the internal
  part→event mapper the run loop applies at its delivery site. Apps never
  map parts themselves — they receive the events via onProgress/attach.
- `src/storage.ts` — built-in `StorageProvider` implementations:
  `fileStorage` (JSONL), `memoryStorage`, and the `serializedStorage`
  wrapper. The interface itself and its documented contract (durable atomic
  append, ordered read-your-writes load, verbatim round-trip via the codec,
  single writer per session per process group, no projection inside append)
  live on the `StorageProvider` JSDoc in `src/types.ts`.
- `src/serialize.ts` — exported binary-safe `StoredEvent` codec used by both
  built-in providers and available to custom storage adapters.
- `src/failure.ts` / `src/backoff.ts` — error classification
  (`transient` | `context-overflow` | `fatal`) and capped exponential backoff.
- `src/compaction.ts` — token-threshold summarization into a hand-off message.
- `src/sanitize.ts` — heals interrupted transcripts (dangling tool calls)
  before replay.
- `src/openrouter.ts` / `src/modelMeta.ts` / `src/usage.ts` — provider
  factory, context-window lookup, per-step usage/cost extraction (see
  "Billing" below).
- `src/resilientFetch.ts` — header + SSE-idle stall detection.
- `src/toolGuard.ts` — caps tool result sizes.
- `src/logEvents.ts` — the default console logger (`createAgentic({ logs:
  true })`): one chalk-colored line per `AgenticEvent`.
- `src/index.ts` — the export surface, grouped by API: the harness
  (`createAgentic` + types), projection (transcripts + live progress),
  storage (providers + event codec), à-la-carte helpers, and an
  advanced/debug section holding only `replaySession`. Loop, backoff,
  compaction, and totals internals are deliberately NOT re-exported —
  consumers get the harness and the read APIs; tests reach into modules.

### Message queueing (the mailbox)

`send()` on a session with a live run queues the message into it: the message
is appended to the ledger FIRST (durable — a crash can never drop it), then a
`RunMailbox` signals the loop, which stops at the next step boundary and folds
the message in. Key invariants, in `src/run.ts`:

- The queue IS storage — the mailbox carries only `queueId`s, never content.
- Pickup is per step, so batched parallel tool calls always complete before a
  queued message is folded in; a tool-call/result pair is never split.
- `hoistPendingInputs()` reorders (per-request only) the exact unanswered
  input messages replay identifies, so a message that landed mid-step is read
  by the model as the newest input; the ledger keeps arrival order.
  `replaySession()` uses each step's `inputQueueIds` to make the same causal
  projection; consumers must not treat raw append order as a ready-to-display
  transcript.
- A run never ends with unanswered input queued; `mailbox.accepting` flips
  false synchronously before `run-end` so a racing `send()` falls back to its
  own run.
- Replay tracks accepted queued inputs by queue id until a durable non-error
  step explicitly records that it saw them. An empty/error step does not
  acknowledge input, except the explicit-cancellation marker, which
  intentionally settles only the causal inputs recorded on that marker. A
  failed/cancelled `run-end` settles its ordinary initiating input but
  preserves queued inputs it never saw, so `resume()`/`interruptedSessions()`
  can recover them.
- The first durable non-error step that records a queue id counts as handling
  it, including a tool-call/result step. If that run later fails, every joined
  caller receives the persisted failure; the input is not automatically
  replayed because doing so could repeat tool side effects.
- Live listeners (`onPart`, `onProgress`, `attach`) are live-only and are not
  replayed from the ledger. A `queue:false` call already waiting on the
  session lock can become the run that answers older queued input; those
  queued callers still receive the durable final result, but may not receive
  that run's historical deltas.
- Every registered run owns a runtime `AbortController`, including the
  pre-start registration window and auto-resume. `agentic.cancel(sessionId,
  reason)` is the deliberate whole-run escape hatch and, unlike an initiating
  caller's disconnect signal, is not suppressed by attached queued callers.
  It terminates only that run; unseen queued inputs remain causally pending,
  collapse into one automatically started successor run, and—once that
  successor is running—can be stopped by a subsequent `cancel()`.

### Message identity (v0.7/v0.8)

Everything in the event stream is covered by ids, minted once at append time
and paired with timestamps:

- User messages carry `id` on their `user-message` event (for queued sends it
  equals `meta.queueId`; pokes and task prompts get ids too).
- Assistant/tool messages are persisted as `StoredMessage` envelopes
  (`{ id, message }`) inside `step` and `compaction` events; tool calls keep
  the AI SDK's `toolCallId` within content.
- Replay returns `SessionMessage` (`{ id, at, message }`) from
  `session.messages()` / `replaySession().messages` — `at` is the persisting
  event's time. Identity survives compaction: a retained tail message keeps
  its original id and `at` in the rebased base (per-message `at` on the
  compaction envelope), and replay re-links pending inputs by id. Internally
  replay threads ids through `sanitizeConversation` via symbol tags so
  clone-on-repair keeps them.
- Live `AgenticEvent`s all carry `at`; the live `step` event includes the
  persisted `StoredMessage[]`, and `queued-message`/`poke` carry `messageId`.
- Steps also carry `inputMessageIds`, the complete durable input membership for
  that model pass. `projectRun(events, runId)` turns those memberships into
  causal response segments for database/UI adapters; app code should not parse
  queue markers or infer turns from append order.
- `RunResult.runId` names the run that produced the result (for a queued send
  answered by another run, the run that causally answered it);
  `RunResult.messageId` is always the send's OWN durable user-message id, even
  when the result is another run's merged answer.
- Identities surface at send time (v0.8): `SendOptions.onAccepted` fires
  exactly once per send — after the durable user-message append, before that
  send's first stream part — with `{ messageId, runId, queued, at }` (queued
  sends report the live run joined; successor collapse can change the
  answering run, so `RunResult.runId`/transcript ids stay authoritative). A
  throwing callback is contained, like `onPart`. Every `onPart` delivery
  carries a `StreamContext` (`{ runId, responseId }`); `responseId` is the
  `${runId}/${segmentIndex}` transcript response item the current model pass
  streams into, derived in the loop from the projected segments plus the
  pass's pending inputs (the same begin-a-segment condition projection
  applies), so it always equals the id `projectSession` later assigns.
- v0.9 requires ledgers written by v0.7 or later. Events carry a schema
  version (`v: 1`), stamped by `encodeEvent`; `decodeEvent` is the single
  validation/normalization point — it stamps well-formed unversioned (v0.7)
  events and rejects pre-v0.7 shapes and newer-than-known versions with
  descriptive errors. Pre-v0.7 session data is unsupported.

`task({ durable: false })` runs against an isolated in-memory ledger. Use it
for disposable presentation work that needs task validation/retries but should
not create a recoverable session in the configured storage.

### Live progress (v0.9)

- The harness emits `ProgressEvent`s itself; apps never map stream parts.
  Delivery surfaces: `SendOptions.onProgress`/`TaskOptions.onProgress` (the
  initiating caller) and `session.attach(listener)` (the transport tail:
  every run this process executes for the session — the currently live run
  and runs that start later, tasks included, `durable: false` ones too).
  Raw stream parts remain available ONLY to the initiating caller via
  `SendOptions.onPart`. All listeners are best-effort/live-only and
  error-contained; detach via attach's returned function. Every entry point
  that goes through the internal `execRun` broadcasts (send, queued
  recovery, resume, auto-resume sweep, and tasks); direct `runLoop()` use
  does not (no instance registry).
- One channel, not two: the loop wraps raw parts and progress events into a
  `RunDelivery` union delivered to the run's options callbacks and the
  mailbox's queue-id-gated listeners (a queued send's onPart/onProgress ride
  one union listener; a queued recovery run inherits them on its mailbox).
  `execRun` composes the attach broadcast over `onProgress` only.
- Emission points (src/run.ts): `response-start` before `streamText` for
  every pass (retries included — same `responseId`) and again at each later
  `start-step` of a tool loop, because each model request's deltas restart
  at offset 0; text/reasoning/tool-start/tool-end derived from parts at the
  delivery site (tool-start from `tool-call`, the one part every provider
  emits exactly once, with parsed input; tool-end from non-preliminary
  `tool-result`); `retry` at both retry-scheduling sites with the attempt;
  `run-end` at all three terminal sites (completed exit, fail(), cancel()).
- The loop mirrors the in-flight pass's accumulated text on `mailbox.partial`
  (nulled the moment the step persists and on run end); `transcript()`
  overlays it as `partialText` on the matching "streaming" response item —
  `projectSession` itself stays pure over the ledger. Text events carry
  `offset` (chars of the pass's partial text before the delta), so a
  reconnecting client attaches first, snapshots `transcript()`, drops text
  events with `offset < partialText.length` — and on every `response-start`
  resets its expected offset to 0 and (re)keys the bubble by `responseId`.
  Snapshot and stream never double-render. partialText/offset dedupe
  semantics are unchanged from v0.8.

### Auto-resume (the boot sweep)

`createAgentic({ autoResume: (sessionId) => agentConfig })` makes crash
recovery automatic: a background sweep runs at creation (and on demand via
`resumeInterrupted()`), finds sessions with an open run or pending queued
messages, and re-drives them. The resolver exists because agent configs hold
tool *functions*, which can't live in storage. Guards: every resume appends a
`run-resume` ledger event (`auto: true` for sweep-driven ones); replay counts
them as `autoResumeAttempts`, and the sweep gives up after `maxAttempts`
(default 3, reset once no queued recovery input remains) so a run that crashes
the process on resume can't wedge a restart loop. Give-up leaves queued-only
work durable for a later manual `resume()`. Sweeps keep at most `maxConcurrent`
(default 4) resumed runs in flight, while `staggerMs` optionally spaces their
starts. Manual `resume()` is uncapped. Sweep kicks re-check under the session
lock, so racing sweeps/sends never double-resume.

Recovery of an interrupted OPEN run is chosen per session by
`autoResume.onInterrupted?: (sessionId) => "resume" | "restart" | "fail"`,
consulted under the session lock after reconciliation (queued-only recovery
never consults it — nothing to discard). `"resume"` (the default, and the
contained fallback when the hook throws/rejects) re-enters the run in place.
`"restart"` — for agents whose tool state lived in process memory and died
with it — closes the dead run with `run-end { status: "failed", discarded:
true }` and re-drives its inputs in a fresh run via the normal queued-recovery
path; it appends the same `run-resume { auto: true }`, so `maxAttempts` caps
crash-looping restarts identically. `"fail"` closes the run plainly (initiating
input settled, unseen queued input preserved), re-drives nothing, and consumes
no attempt. Replay handles `discarded` by snapshot-rewind: `run-start`
snapshots the working state, and a discarded run-end restores it plus the
non-poke user messages that arrived during the run — to messages AND pending,
because a discarded run's acknowledgments are void (pokes are not restored;
their purpose died with the run). Mid-run compaction inside a discarded run
vanishes with the rewind; usage totals do not (the cost was real). `runLoop`
never writes `discarded` — only the sweep's restart path does.

### Billing (BYOK vs credits)

Cost extraction (`src/usage.ts`) is `is_byok`-aware. OpenRouter usage
accounting reports `cost` plus `cost_details.upstream_inference_cost`, and
which one is the real charge depends on the billing regime — since mid-2026
the upstream figure is populated on credits-paid requests too, as a mirror
of what OpenRouter paid the provider (informational, not an extra charge),
so the field's presence no longer identifies BYOK. `reconcileBilledCost`
branches on OpenRouter's `is_byok` flag: BYOK → sum (`cost` is the fee,
often 0; upstream is the provider bill); credits → `cost` alone (summing
would double-count the mirror); flag absent → never sum, take `cost` falling
back to upstream, because doubling a mirrored credits charge is worse than
undercounting a fee-only BYOK payload. `is_byok` only survives on the raw
snake_case usage payload (`usage.raw`) — `@openrouter/ai-sdk-provider`
(≤2.3.3) drops it from the camelCase `providerMetadata.openrouter.usage` —
so `extractStepUsage` reads both shapes and keeps all three numbers on
`StepUsage` (`cost`, `upstreamCost`, `isByok`, reconciled `billedCost`).
Totals accumulate `billedCost`, never raw `cost`.

## Testing

Tests live next to source (`src/*.test.ts` — harness, transcript, attach,
projection, storage, modelMeta, openrouter) and run fully offline: model calls use
`MockLanguageModelV3` from `ai/test` with hand-built `LanguageModelV3StreamPart`
arrays, storage uses `memoryStorage()`, and context windows are pinned with
`setContextWindow("mock/model", …)`. No API keys needed.

`playground/boilerplate.ts` is the copy-me starting point (`npm run
playground`); `playground/mvp/` has runnable end-to-end proof demos. Both
need `OPENROUTER_API_KEY` in `.env` — see README. Demo session data
(`playground/mvp/.queue-demo/` etc.) is gitignored.

## Releasing

1. Bump `version` in `package.json` (semver).
2. Commit, push.
3. `npm publish` — `prepublishOnly` runs the build; package ships `dist/`
   only, with `publishConfig.access: public`.
