// Level 0 — drop-in OpenRouter provider.

// Level 2 — the harness: durable sessions, guaranteed-outcome tasks, storage.
export {
	type Agentic,
	type AgenticOptions,
	type AutoResumeOptions,
	type AutoResumeResolver,
	createAgentic,
	type SendOptions,
	type Session,
	type TaskOptions,
	type WithRetriesOptions,
	withRetries,
} from "./agentic.js";
export { retryDelayMs, wait } from "./backoff.js";
export {
	DEFAULT_COMPACTION_PROMPT,
	runCompaction,
	shouldCompact,
} from "./compaction.js";
// Level 1 — à-la-carte helpers, usable with plain streamText/generateText.
export { classifyFailure, describeError, isContextOverflow, serializeError } from "./failure.js";
export { logEvents } from "./logEvents.js";
export { getContextWindow, setContextWindow } from "./modelMeta.js";
export {
	createOpenRouter,
	type OpenRouterProvider,
	type OpenRouterProviderSettings,
} from "./openrouter.js";
export { type ProgressEvent, progressFromPart } from "./progress.js";
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
	type TranscriptItem,
	type TranscriptResponseItem,
	type TranscriptUserItem,
	textOf,
} from "./projection.js";
export { type ReplayedSession, replaySession } from "./replay.js";
export {
	createResilientFetch,
	HeaderTimeoutError,
	type ResilientFetchOptions,
	StreamIdleTimeoutError,
} from "./resilientFetch.js";
export { createMailbox, type RunLoopOptions, type RunMailbox, runLoop } from "./run.js";
export { sanitizeConversation } from "./sanitize.js";
export { decodeEvent, encodeEvent } from "./serialize.js";
export { fileStorage, memoryStorage, serializedStorage } from "./storage.js";
export { guardToolResultSizes, type ToolGuardOptions } from "./toolGuard.js";
export type {
	AgentConfig,
	AgenticEvent,
	ClassifiedFailure,
	CompactionConfig,
	EventListener,
	FailureKind,
	MaybePromise,
	RetryConfig,
	RunResult,
	SerializedError,
	SessionMessage,
	StepUsage,
	StorageProvider,
	StoredEvent,
	StoredMessage,
	TaskOutcome,
	UsageTotals,
} from "./types.js";
export {
	addStepToTotals,
	contextTokensOf,
	emptyTotals,
	extractStepUsage,
	reconcileBilledCost,
} from "./usage.js";
