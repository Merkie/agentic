/** biome-ignore-all assist/source/organizeImports: exports are grouped by API surface, not sorted by module */

// ── the harness: durable sessions, guaranteed-outcome tasks, auto-resume ──
export {
	type Agentic,
	type AgenticOptions,
	type AutoResumeOptions,
	type AutoResumeResolver,
	createAgentic,
	type SendOptions,
	type Session,
	type TaskOptions,
} from "./agentic.js";
export type {
	AgentConfig,
	AgenticEvent,
	CompactionConfig,
	EventListener,
	RetryConfig,
	RunResult,
	SessionMessage,
	StepUsage,
	TaskOutcome,
	UsageTotals,
} from "./types.js";

// ── projection: the read APIs — transcripts for UIs, live progress for wires ──
export {
	type ProjectedInput,
	type ProjectedRun,
	type ProjectedRunSegment,
	projectRun,
	projectSession,
	type ResponseStatus,
	type RunProjectionStatus,
	type SessionTranscript,
	type StreamContext,
	textOf,
	type TranscriptItem,
	type TranscriptResponseItem,
	type TranscriptUserItem,
} from "./projection.js";
export { type ProgressEvent, progressFromPart } from "./progress.js";

// ── storage: bring-your-own persistence + the event codec ────────────────
export { decodeEvent, encodeEvent } from "./serialize.js";
export { fileStorage, memoryStorage, serializedStorage } from "./storage.js";
export type { StorageProvider, StoredEvent, StoredMessage } from "./types.js";

// ── à-la-carte helpers, usable with plain streamText/generateText ────────
export { type WithRetriesOptions, withRetries } from "./agentic.js";
export { classifyFailure, describeError, isContextOverflow, serializeError } from "./failure.js";
export { logEvents } from "./logEvents.js";
export { getContextWindow, setContextWindow } from "./modelMeta.js";
export {
	createOpenRouter,
	type OpenRouterProvider,
	type OpenRouterProviderSettings,
} from "./openrouter.js";
export {
	createResilientFetch,
	HeaderTimeoutError,
	type ResilientFetchOptions,
	StreamIdleTimeoutError,
} from "./resilientFetch.js";
export { sanitizeConversation } from "./sanitize.js";
export { guardToolResultSizes, type ToolGuardOptions } from "./toolGuard.js";
export type { ClassifiedFailure, FailureKind, SerializedError } from "./types.js";
export { extractStepUsage, reconcileBilledCost } from "./usage.js";

// ── advanced/debug: raw ledger replay ────────────────────────────────────
// The default read path is session.transcript() / projectSession. replaySession
// rebuilds the MODEL's view (replay-ready messages + run state) straight from
// events — for debugging and tooling, not for rendering a chat.
export { type ReplayedSession, replaySession } from "./replay.js";
