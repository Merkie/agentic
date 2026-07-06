import type { ModelMessage, StopCondition, ToolSet } from "ai";

export type MaybePromise<T> = T | Promise<T>;

// ── usage & cost ────────────────────────────────────────────────────────

/**
 * Normalized per-step usage. `cost` is what OpenRouter charged in credits;
 * `upstreamCost` is the underlying provider charge (populated on BYOK where
 * `cost` is just OpenRouter's fee or 0). `billedCost` is the reconciled
 * "what this actually cost me" number — see {@link reconcileBilledCost}.
 */
export interface StepUsage {
	inputTokens: number | null;
	outputTokens: number | null;
	totalTokens: number | null;
	cachedInputTokens: number | null;
	reasoningTokens: number | null;
	cost: number | null;
	upstreamCost: number | null;
	billedCost: number | null;
}

export interface UsageTotals {
	inputTokens: number;
	outputTokens: number;
	totalTokens: number;
	cachedInputTokens: number;
	reasoningTokens: number;
	cost: number;
	steps: number;
	/** Last step's input+output tokens — the live conversation size. */
	contextTokens: number | null;
	/** Model context window, when known. */
	contextWindow: number | null;
}

// ── storage ─────────────────────────────────────────────────────────────

export interface SerializedError {
	name?: string;
	message: string;
	status?: number | string;
	detail?: unknown;
}

/**
 * The append-only event ledger. One session = one ordered list of events.
 * Replaying the list rebuilds the exact ModelMessage[] the model sees plus
 * run/audit state — this is what makes runs survive process restarts.
 */
export type StoredEvent =
	| {
			type: "user-message";
			at: string;
			message: ModelMessage;
			/** App-supplied tag (e.g. a poke dedup key); opaque to the framework. */
			meta?: Record<string, unknown>;
	  }
	| { type: "run-start"; at: string; runId: string; model: string }
	| {
			type: "step";
			at: string;
			runId: string;
			/** The step's response messages (assistant + tool), replay-ready. */
			messages: ModelMessage[];
			finishReason: string;
			usage: StepUsage;
	  }
	| {
			type: "compaction";
			at: string;
			/** The new replay base — everything before this event is superseded. */
			messages: ModelMessage[];
			usage?: StepUsage;
	  }
	| {
			type: "run-end";
			at: string;
			runId: string;
			status: "completed" | "cancelled" | "failed";
			error?: SerializedError;
	  };

/**
 * Bring-your-own persistence. All the framework needs is append + load in
 * order. Implementations may be sync or async (file, SQLite, Prisma, Redis…).
 */
export interface StorageProvider {
	append(sessionId: string, event: StoredEvent): MaybePromise<void>;
	load(sessionId: string): MaybePromise<StoredEvent[]>;
	/** Optional: session ids with data, for boot-time resume sweeps. */
	listSessions?(): MaybePromise<string[]>;
}

// ── failure classification ──────────────────────────────────────────────

export type FailureKind =
	/** Worth retrying with backoff (5xx, 429, socket drops, stream stalls…). */
	| "transient"
	/** Context window overflow — a compaction signal, never a retry. */
	| "context-overflow"
	/** Deterministic (auth, billing, policy, malformed request). Never retry. */
	| "fatal";

export interface ClassifiedFailure {
	kind: FailureKind;
	/** Server-directed wait (Retry-After), when the error carried one. */
	retryAfterMs?: number;
	error: SerializedError;
}

// ── agent & run configuration ───────────────────────────────────────────

export interface CompactionConfig {
	/**
	 * Compact when the last step's context usage exceeds this fraction of the
	 * model's context window (from OpenRouter model metadata). 0 disables.
	 */
	limit?: number;
	/** Messages kept verbatim after the summary. */
	keepRecent?: number;
	/** Override the summarizer system prompt. */
	prompt?: string;
	/** Summarize on a different (cheaper) model. Defaults to the agent model. */
	model?: string;
}

export interface RetryConfig {
	/** Max transient retries without progress before the run fails. */
	maxAttempts?: number;
	/** Base backoff delay (ms); doubles per attempt. */
	baseDelayMs?: number;
	/** Backoff ceiling (ms). Retry-After from the server wins over backoff. */
	maxDelayMs?: number;
	/** Hard wall-clock budget (ms) across all retries of one run. */
	maxElapsedMs?: number;
}

export interface AgentConfig<TOOLS extends ToolSet = ToolSet> {
	/** OpenRouter model id, e.g. "qwen/qwen3.7-max". */
	model: string;
	system?: string;
	tools?: TOOLS;
	/** Max model steps per run. */
	maxSteps?: number;
	/** Extra stop conditions merged with the framework's. */
	stopWhen?: StopCondition<NoInfer<TOOLS>>[];
	compaction?: CompactionConfig;
	retry?: RetryConfig;
	/**
	 * Cap serialized tool-result size so no single result can blow the context
	 * window. Defaults to truncate at 120 KB; `false` disables.
	 */
	toolResultGuard?: { maxBytes?: number; mode?: "truncate" | "discard" } | false;
	/** Keep providerOptions (reasoning signatures etc.) on persisted messages. */
	preserveProviderOptions?: boolean;
	/** OpenRouter provider routing etc., merged into extraBody per call. */
	extraBody?: Record<string, unknown>;
}

// ── observability ───────────────────────────────────────────────────────

export type AgenticEvent =
	| { type: "run-start"; sessionId: string; runId: string; model: string }
	| {
			type: "step";
			sessionId: string;
			runId: string;
			finishReason: string;
			usage: StepUsage;
			toolCalls: { toolName: string; input: unknown }[];
			text: string;
	  }
	| {
			type: "retry";
			sessionId: string;
			runId: string;
			attempt: number;
			maxAttempts: number;
			delayMs: number;
			error: SerializedError;
	  }
	| { type: "compaction"; sessionId: string; beforeTokens: number | null; summaryChars: number }
	| { type: "poke"; sessionId: string; runId: string; poke: number; maxPokes: number }
	/** A send() landed while a run was live and was queued into it. */
	| { type: "queued-message"; sessionId: string; runId: string | null }
	| {
			type: "run-end";
			sessionId: string;
			runId: string;
			status: "completed" | "cancelled" | "failed";
			totals: UsageTotals;
			error?: SerializedError;
	  };

export type EventListener = (event: AgenticEvent) => void;

// ── results ─────────────────────────────────────────────────────────────

export interface RunResult {
	status: "completed" | "cancelled" | "failed";
	/** Text of the final assistant message (all text parts joined). */
	text: string;
	totals: UsageTotals;
	error?: SerializedError;
}

export type TaskOutcome<T = unknown> =
	| { status: "submitted"; deliverable: T; totals: UsageTotals; sessionId: string }
	| { status: "cancelled"; reason: string; totals: UsageTotals; sessionId: string }
	| { status: "failed"; error: SerializedError; totals: UsageTotals; sessionId: string };
