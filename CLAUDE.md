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
  chat: `send`/`resume`/`messages`/`isInterrupted`), `task()`
  (guaranteed-outcome workflows via `submit_deliverable`/`cancel_task`
  tools), `withRetries()`. Holds per-session locks and the live-run map.
- `src/run.ts` — `runLoop()`, the heart: replay ledger → streamText → persist
  steps → repeat. Owns retries (via `classifyFailure`), mid-run compaction,
  poking, and the `RunMailbox` for live message queueing.
- `src/replay.ts` — `replaySession(events)`: ledger → messages + usage
  totals + `interruptedRunId` + `pendingMessages`.
- `src/storage.ts` — `StorageProvider` interface (`append`/`load`, optional
  `listSessions`); `fileStorage` (JSONL) and `memoryStorage` built in.
- `src/failure.ts` / `src/backoff.ts` — error classification
  (`transient` | `context-overflow` | `fatal`) and capped exponential backoff.
- `src/compaction.ts` — token-threshold summarization into a hand-off message.
- `src/sanitize.ts` — heals interrupted transcripts (dangling tool calls)
  before replay.
- `src/openrouter.ts` / `src/modelMeta.ts` / `src/usage.ts` — provider
  factory, context-window lookup, per-step usage/cost extraction
  (BYOK-aware).
- `src/resilientFetch.ts` — header + SSE-idle stall detection.
- `src/toolGuard.ts` — caps tool result sizes.
- `src/logStream.ts` — pretty stream printer for demos/debugging.
- `src/index.ts` — the export surface, organized in levels: Level 0
  (provider + logging), Level 1 (à-la-carte helpers), Level 2 (the harness).

### Message queueing (the mailbox)

`send()` on a session with a live run queues the message into it: the message
is appended to the ledger FIRST (durable — a crash can never drop it), then a
`RunMailbox` signals the loop, which stops at the next step boundary and folds
the message in. Key invariants, in `src/run.ts`:

- The queue IS storage — the mailbox carries only `queueId`s, never content.
- Pickup is per step, so batched parallel tool calls always complete before a
  queued message is folded in; a tool-call/result pair is never split.
- `hoistSandwichedUsers()` reorders (per-request only) a message that landed
  mid-step so the model reads it as the newest input; the ledger keeps
  arrival order.
- A run never ends with unanswered input queued; `mailbox.accepting` flips
  false synchronously before `run-end` so a racing `send()` falls back to its
  own run.
- Replay counts trailing unanswered messages (`pendingMessages`) so
  `resume()`/`interruptedSessions()` recover queued messages orphaned by a
  crash — but any `run-end` clears them (don't auto-re-run an abort/failure).

## Testing

Tests live next to source (`src/harness.test.ts`, `src/logStream.test.ts`,
`src/openrouter.test.ts`) and run fully offline: model calls use
`MockLanguageModelV3` from `ai/test` with hand-built `LanguageModelV3StreamPart`
arrays, storage uses `memoryStorage()`, and context windows are pinned with
`setContextWindow("mock/model", …)`. No API keys needed.

`playground/mvp/` has runnable end-to-end proof demos (need
`OPENROUTER_API_KEY` in `.env`) — see README. Demo session data
(`playground/mvp/.queue-demo/` etc.) is gitignored.

## Releasing

1. Bump `version` in `package.json` (semver).
2. Commit, push.
3. `npm publish` — `prepublishOnly` runs the build; package ships `dist/`
   only, with `publishConfig.access: public`.
