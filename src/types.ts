import type { ModelMessage, StopCondition, ToolChoice, ToolSet } from "ai";

export type MaybePromise<T> = T | Promise<T>;

// ── usage & cost ────────────────────────────────────────────────────────

/**
 * Normalized per-step usage. `cost` is what OpenRouter charged in credits
 * (on BYOK requests: just OpenRouter's fee, often 0). `upstreamCost` is the
 * underlying provider charge — on BYOK it's the real bill; on credits
 * requests OpenRouter mirrors what it paid the provider here, so it is
 * informational, NOT an extra charge. `isByok` is OpenRouter's discriminator
 * between the two regimes (null when not reported). `billedCost` is the
 * reconciled "what this actually cost me" number — see
 * {@link reconcileBilledCost}. Only `billedCost` belongs in spend totals.
 */
export interface StepUsage {
	inputTokens: number | null;
	outputTokens: number | null;
	totalTokens: number | null;
	cachedInputTokens: number | null;
	reasoningTokens: number | null;
	cost: number | null;
	upstreamCost: number | null;
	isByok: boolean | null;
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
 * Ledger form of a conversation message: the model-ready payload plus its
 * framework-generated identity. Ids are minted once, at append time, and are
 * stable for the life of the session — compaction carries them (and the
 * original `at`) into the rebased message base unchanged.
 */
export interface StoredMessage {
	id: string;
	/**
	 * Original persist time, when it differs from the carrying event's `at`
	 * (compaction re-bases old messages under a new event). Absent ⇒ the
	 * event's `at` is the message's time.
	 */
	at?: string;
	message: ModelMessage;
}

/**
 * Replayed form of a conversation message: what {@link Session.messages} and
 * `replaySession().messages` return.
 */
export interface SessionMessage {
	id: string;
	/** ISO 8601 time the message was persisted. */
	at: string;
	message: ModelMessage;
}

/**
 * The append-only event ledger. One session = one ordered list of events.
 * Replaying the list rebuilds the exact ModelMessage[] the model sees plus
 * run/audit state — this is what makes runs survive process restarts.
 *
 * `v` is the event schema version, stamped by `encodeEvent`. `decodeEvent`
 * is the single validation/normalization point: it stamps well-formed
 * unversioned events (v0.7 predates the field) and rejects pre-v0.7 shapes
 * (missing message identity) and versions from newer agentic releases.
 */
export type StoredEvent =
	| {
			type: "user-message";
			at: string;
			message: ModelMessage;
			v?: 1;
			/**
			 * Framework-generated message identity, minted at append time. For
			 * queued sends this equals `meta.queueId`.
			 */
			id: string;
			/** App-supplied tag (e.g. a poke dedup key); opaque to the framework. */
			meta?: Record<string, unknown>;
	  }
	| { type: "run-start"; at: string; v?: 1; runId: string; model: string }
	| {
			type: "step";
			at: string;
			v?: 1;
			runId: string;
			/**
			 * The step's response messages (assistant + tool), replay-ready, each
			 * carrying its minted id.
			 */
			messages: StoredMessage[];
			/**
			 * Durable ids of the user messages present in the model input that
			 * produced this step — both the initiating input and messages accepted
			 * by a live run. Projection adapters can use it to reconstruct causal
			 * turns without reverse-engineering ledger order.
			 */
			inputMessageIds: string[];
			/**
			 * Queued user-message ids present in the model input that produced this
			 * step. An empty array is meaningful: this step did not see a queued
			 * message that may have arrived while it was in flight.
			 */
			inputQueueIds: string[];
			/**
			 * False when the provider ended the step before producing a durable
			 * response. Such a step records usage/error diagnostics but does not
			 * settle any input; replay and recovery must keep that input pending.
			 */
			acknowledgesInput: boolean;
			finishReason: string;
			usage: StepUsage;
			/**
			 * Terminal intent captured with the step itself. Explicit cancellation
			 * writes this before run-end so a crash in that two-append window is
			 * reconciled as cancelled instead of being resumed or misreported as a
			 * completed answer.
			 */
			interrupted?: {
				status: "cancelled";
				error: SerializedError;
			};
	  }
	| {
			type: "compaction";
			at: string;
			v?: 1;
			/**
			 * The new replay base — everything before this event is superseded.
			 * Messages retained from before the compaction keep their original id
			 * and `at`; the summary message gets a fresh id.
			 */
			messages: StoredMessage[];
			usage?: StepUsage;
	  }
	| {
			/** A resume re-entered interrupted work (open run or queued messages). */
			type: "run-resume";
			at: string;
			v?: 1;
			/** The interrupted run being re-entered; null when resuming queued messages only. */
			runId: string | null;
			/** True when the boot-time auto-resume sweep kicked it (counts toward the attempt cap). */
			auto?: boolean;
	  }
	| {
			type: "run-end";
			at: string;
			v?: 1;
			runId: string;
			status: "completed" | "cancelled" | "failed";
			/** Reconciliation close: do not settle inputs appended after this run's final step. */
			preservePending?: boolean;
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
	/** AI SDK tool-selection policy (`auto`, `required`, `none`, or one named tool). */
	toolChoice?: ToolChoice<NoInfer<TOOLS>>;
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
	| { type: "run-start"; at: string; sessionId: string; runId: string; model: string }
	| {
			type: "step";
			at: string;
			sessionId: string;
			runId: string;
			finishReason: string;
			usage: StepUsage;
			/** The step's persisted messages with their minted ids — what the ledger recorded. */
			messages: StoredMessage[];
			/** User-message ids present in the model input for this persisted step. */
			inputMessageIds: string[];
			toolCalls: { toolName: string; input: unknown }[];
			text: string;
	  }
	| {
			type: "retry";
			at: string;
			sessionId: string;
			runId: string;
			attempt: number;
			maxAttempts: number;
			delayMs: number;
			error: SerializedError;
	  }
	| {
			type: "compaction";
			at: string;
			sessionId: string;
			beforeTokens: number | null;
			summaryChars: number;
	  }
	| {
			type: "poke";
			at: string;
			sessionId: string;
			runId: string;
			poke: number;
			maxPokes: number;
			/** Id of the poke user-message appended to the ledger. */
			messageId: string;
	  }
	/** A send() landed while a run was live and was queued into it. */
	| {
			type: "queued-message";
			at: string;
			sessionId: string;
			runId: string | null;
			/** Id of the queued user-message (also its queueId). */
			messageId: string;
	  }
	/** The auto-resume sweep found interrupted work: resuming it, or giving up after too many attempts. */
	| {
			type: "auto-resume";
			at: string;
			sessionId: string;
			runId: string | null;
			attempt: number;
			maxAttempts: number;
			action: "resume" | "give-up";
	  }
	| {
			type: "run-end";
			at: string;
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
	/** The run that produced this result — correlates with run/step events and step messages. */
	runId: string;
	/** Text of the final assistant message (all text parts joined). */
	text: string;
	totals: UsageTotals;
	error?: SerializedError;
}

export type TaskOutcome<T = unknown> =
	| { status: "submitted"; deliverable: T; totals: UsageTotals; sessionId: string }
	| { status: "cancelled"; reason: string; totals: UsageTotals; sessionId: string }
	| { status: "failed"; error: SerializedError; totals: UsageTotals; sessionId: string };
