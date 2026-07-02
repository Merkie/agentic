// Level 0 — drop-in OpenRouter provider + stream pretty-printer.
export {
	createOpenRouter,
	type OpenRouterProvider,
	type OpenRouterProviderSettings,
} from "./openrouter.js";
export { logStream } from "./logStream.js";

// Level 1 — à-la-carte helpers, usable with plain streamText/generateText.
export { classifyFailure, describeError, isContextOverflow, serializeError } from "./failure.js";
export {
	createResilientFetch,
	HeaderTimeoutError,
	StreamIdleTimeoutError,
	type ResilientFetchOptions,
} from "./resilientFetch.js";
export { sanitizeConversation } from "./sanitize.js";
export { guardToolResultSizes, type ToolGuardOptions } from "./toolGuard.js";
export {
	addStepToTotals,
	contextTokensOf,
	emptyTotals,
	extractStepUsage,
	reconcileBilledCost,
} from "./usage.js";
export { getContextWindow, setContextWindow } from "./modelMeta.js";
export { retryDelayMs, wait } from "./backoff.js";
export { withRetries, type WithRetriesOptions } from "./agentic.js";

// Level 2 — the harness: durable sessions, guaranteed-outcome tasks, storage.
export {
	createAgentic,
	type Agentic,
	type AgenticOptions,
	type SendOptions,
	type Session,
	type TaskOptions,
} from "./agentic.js";
export { fileStorage, memoryStorage } from "./storage.js";
export { replaySession, type ReplayedSession } from "./replay.js";
export { runLoop, type RunLoopOptions } from "./run.js";
export {
	DEFAULT_COMPACTION_PROMPT,
	runCompaction,
	shouldCompact,
} from "./compaction.js";

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
	StepUsage,
	StorageProvider,
	StoredEvent,
	TaskOutcome,
	UsageTotals,
} from "./types.js";
