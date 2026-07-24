import {
	type LanguageModel,
	type ModelMessage,
	stepCountIs,
	streamText,
	type TextStreamPart,
	type ToolCallPart,
	type ToolResultPart,
	type ToolSet,
} from "ai";
import { resolveRetryConfig, retryDelayMs, wait } from "./backoff.js";
import { resolveCompactionConfig, runCompaction, shouldCompact } from "./compaction.js";
import { classifyFailure, serializeError } from "./failure.js";
import { getContextWindow } from "./modelMeta.js";
import { projectRun, type StreamContext } from "./projection.js";
import { replaySession } from "./replay.js";
import { guardToolResultSizes } from "./toolGuard.js";
import type {
	AgentConfig,
	AgenticEvent,
	EventListener,
	RunResult,
	SessionMessage,
	StepUsage,
	StorageProvider,
	StoredEvent,
	StoredMessage,
	UsageTotals,
} from "./types.js";
import { addStepToTotals, contextTokensOf, emptyTotals, extractStepUsage } from "./usage.js";

// The agent loop. One invocation = one "run" that is guaranteed to terminate
// in a run-end event, no matter how many transient provider failures, stream
// drops, compactions, or pokes happen along the way.
//
// The loop is STATELESS OVER STORAGE: every pass replays the session from the
// event ledger, so a run interrupted by a process kill resumes from its last
// finished step — re-entering this function with the same sessionId is the
// entire recovery story. Steps persist the moment they finish streaming
// (onStepFinish), so a 30-minute run that dies at minute 29 loses at most the
// step in flight.

// ── message queueing ─────────────────────────────────────────────────────
// A mailbox is the in-process half of message queueing (the design is
// opencode's: THE QUEUE IS STORAGE). send() on a busy session appends the
// user message to the ledger first — durable immediately, so a crash can
// never drop it — then signals the live run through its mailbox so the loop
// picks the message up at the next step boundary instead of ending. The
// ledger alone also recovers the cross-process/crash cases: replay counts
// trailing unanswered user messages (`pendingMessages`) so resume() knows
// there is queued work even when the mailbox died with its process.

export interface RunMailbox {
	/** The live run's id, set when the loop starts. */
	runId: string | null;
	/**
	 * False once the run has committed to ending. tryEnqueue() then refuses,
	 * telling the caller to start a fresh run for its (already-persisted)
	 * message instead.
	 */
	accepting: boolean;
	/** Durable callers attached to this shared run, including append-in-flight sends. */
	attachedQueueIds: Set<string>;
	/** queueIds of ledger-appended messages the loop has not yet consumed. */
	queued: string[];
	/** Live-stream listeners contributed by attached send() calls. */
	partListeners: Array<(part: unknown, context: StreamContext) => void>;
	/** Queue-id-gated listeners; activated only once a model pass includes that input. */
	queuedPartListeners: Array<{
		queueId: string;
		listener: (part: unknown, context: StreamContext) => void;
		active: boolean;
	}>;
	/**
	 * Send-acceptance callbacks waiting for this mailbox's run id. Registered
	 * while `runId` is still null (a send joined a run registered but not yet
	 * started); execRun flushes them the moment the loop has assigned the id —
	 * before any stream part can be delivered — and a recovery run inherits
	 * whatever the dead run never flushed.
	 */
	acceptHooks: Array<(runId: string) => void>;
	/** Signal a new ledger message. Returns false if the run stopped listening. */
	tryEnqueue(queueId: string): boolean;
}

export function createMailbox(): RunMailbox {
	const mailbox: RunMailbox = {
		runId: null,
		accepting: true,
		attachedQueueIds: new Set(),
		queued: [],
		partListeners: [],
		queuedPartListeners: [],
		acceptHooks: [],
		tryEnqueue(queueId: string) {
			if (!mailbox.accepting) return false;
			mailbox.attachedQueueIds.add(queueId);
			mailbox.queued.push(queueId);
			return true;
		},
	};
	return mailbox;
}

function ledgerHasQueued(events: StoredEvent[], queueId: string): boolean {
	for (let i = events.length - 1; i >= 0; i--) {
		const event = events[i];
		if (
			event.type === "user-message" &&
			event.meta?.queued === true &&
			event.meta.queueId === queueId
		) {
			return true;
		}
	}
	return false;
}

// A queued message that arrived while a step was streaming sits in the
// ledger BEFORE that step's event, so the replayed conversation shows it
// sandwiched under model output that never saw it. For the next request,
// move that trailing user block after the model's output so the model reads
// it as the newest input. Per-request only — the ledger keeps arrival order.
// Exported for tests: a splice off-by-one here would wedge a user message
// between a tool call and its result, which providers reject outright.
export function hoistSandwichedUsers(messages: ModelMessage[], max: number): void {
	if (messages.length === 0 || max <= 0) return;
	const queued: ModelMessage[] = [];
	for (let index = messages.length - 1; index >= 0 && queued.length < max; index--) {
		if (messages[index].role !== "user") continue;
		queued.unshift(...messages.splice(index, 1));
	}
	messages.push(...queued);
}

function hoistPendingInputs(messages: SessionMessage[], pending: SessionMessage[]): void {
	const actionable: SessionMessage[] = [];
	for (const message of pending) {
		const index = messages.indexOf(message);
		if (index >= 0) messages.splice(index, 1);
		actionable.push(message);
	}
	messages.push(...actionable);
}

export interface RunLoopOptions<TOOLS extends ToolSet = ToolSet> {
	sessionId: string;
	agent: AgentConfig<TOOLS>;
	storage: StorageProvider;
	/** Model factory (modelId → LanguageModel), wired to OpenRouter upstream. */
	// biome-ignore lint/suspicious/noExplicitAny: contravariant position — any AgentConfig works
	getModel: (modelId: string, agent: AgentConfig<any>) => LanguageModel;
	emit?: EventListener;
	abortSignal?: AbortSignal;
	/** Live stream parts (text deltas, tool calls…) for UIs, with their agentic identity. */
	onPart?: (part: TextStreamPart<TOOLS>, context: StreamContext) => void;
	/**
	 * The durable user-message id RunResult.messageId reports as this run's
	 * own. When absent (direct runLoop use), the newest pending input at the
	 * first pass is used.
	 */
	messageId?: string;
	/**
	 * Workflow support: when set, a pass that ends cleanly while
	 * `isSettled()` is false triggers a poke — `pokeMessage(n)` is appended as
	 * a user message and the loop re-enters, up to `maxPokes` times.
	 */
	isSettled?: () => boolean;
	pokeMessage?: (poke: number) => string;
	maxPokes?: number;
	/** Resume an interrupted run under its original runId. */
	resumeRunId?: string;
	/**
	 * Live message queueing: signals ledger-appended user messages so the loop
	 * picks them up at the next step boundary instead of ending the run.
	 */
	mailbox?: RunMailbox;
}

function newId(): string {
	return globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2);
}

/** An AgenticEvent as constructed at an emit site — the wrapper stamps `at`. */
type EmittableEvent = AgenticEvent extends infer E
	? E extends AgenticEvent
		? Omit<E, "at">
		: never
	: never;

function storeMessages(messages: ModelMessage[]): StoredMessage[] {
	return messages.map((message) => ({ id: newId(), message }));
}

function now(): string {
	return new Date().toISOString();
}

// Persisted messages default to having providerOptions stripped: OpenRouter
// mirrors full reasoning_details onto every part, which multiplies storage
// for replay data the next call doesn't need.
function stripProviderOptions(messages: ModelMessage[]): ModelMessage[] {
	return messages.map((message) => {
		const m = { ...message } as Record<string, unknown>;
		delete m.providerOptions;
		if (Array.isArray(m.content)) {
			m.content = (m.content as Record<string, unknown>[]).map((part) => {
				if (part && typeof part === "object" && "providerOptions" in part) {
					const p = { ...part };
					delete p.providerOptions;
					return p;
				}
				return part;
			});
		}
		return m as ModelMessage;
	});
}

type CapturedToolCall = Pick<
	ToolCallPart,
	"type" | "toolCallId" | "toolName" | "input" | "providerExecuted" | "providerOptions"
>;

type CapturedToolResult = {
	toolCallId: string;
	toolName: string;
	input: unknown;
	kind: "result" | "error" | "denied";
	value?: unknown;
};

interface PartialStepCapture {
	text: string;
	toolCalls: CapturedToolCall[];
	toolResults: Map<string, CapturedToolResult>;
}

function emptyPartialStepCapture(): PartialStepCapture {
	return { text: "", toolCalls: [], toolResults: new Map() };
}

function resetPartialStepCapture(capture: PartialStepCapture): void {
	capture.text = "";
	capture.toolCalls = [];
	capture.toolResults.clear();
}

function captureStreamPart<TOOLS extends ToolSet>(
	capture: PartialStepCapture,
	part: TextStreamPart<TOOLS>,
): void {
	switch (part.type) {
		case "start-step":
		case "finish-step":
			resetPartialStepCapture(capture);
			break;
		case "text-delta":
			capture.text += part.text;
			break;
		case "tool-call":
			if (!capture.toolCalls.some((call) => call.toolCallId === part.toolCallId)) {
				capture.toolCalls.push({
					type: "tool-call",
					toolCallId: part.toolCallId,
					toolName: part.toolName,
					input: part.input,
					...(part.providerExecuted === undefined
						? {}
						: { providerExecuted: part.providerExecuted }),
					...(part.providerMetadata === undefined
						? {}
						: { providerOptions: part.providerMetadata }),
				});
			}
			break;
		case "tool-result":
			// AI SDK tools can emit replaceable previews before their authoritative
			// result. If cancellation lands between the two, treating the preview as
			// final would make replay claim interrupted work completed successfully.
			if (part.preliminary === true) break;
			capture.toolResults.set(part.toolCallId, {
				toolCallId: part.toolCallId,
				toolName: part.toolName,
				input: part.input,
				kind: "result",
				value: part.output,
			});
			break;
		case "tool-error":
			capture.toolResults.set(part.toolCallId, {
				toolCallId: part.toolCallId,
				toolName: part.toolName,
				input: part.input,
				kind: "error",
				value: part.error,
			});
			break;
		case "tool-output-denied": {
			const call = capture.toolCalls.find((candidate) => candidate.toolCallId === part.toolCallId);
			capture.toolResults.set(part.toolCallId, {
				toolCallId: part.toolCallId,
				toolName: part.toolName,
				input: call?.input,
				kind: "denied",
			});
			break;
		}
		case "tool-approval-request":
			if (!capture.toolCalls.some((call) => call.toolCallId === part.toolCall.toolCallId)) {
				capture.toolCalls.push({
					type: "tool-call",
					toolCallId: part.toolCall.toolCallId,
					toolName: part.toolCall.toolName,
					input: part.toolCall.input,
					...(part.toolCall.providerExecuted === undefined
						? {}
						: { providerExecuted: part.toolCall.providerExecuted }),
					...(part.toolCall.providerMetadata === undefined
						? {}
						: { providerOptions: part.toolCall.providerMetadata }),
				});
			}
			break;
	}
}

function safeJsonValue(value: unknown): unknown {
	try {
		const encoded = JSON.stringify(value);
		return encoded === undefined ? String(value) : JSON.parse(encoded);
	} catch {
		return String(value);
	}
}

async function capturedToolOutput(
	result: CapturedToolResult,
	call: CapturedToolCall,
	tools: ToolSet | undefined,
): Promise<ToolResultPart["output"]> {
	if (result.kind === "denied") {
		return { type: "execution-denied", reason: "Tool execution was denied before cancellation." };
	}
	if (result.kind === "error") {
		return { type: "error-text", value: serializeError(result.value).message };
	}
	const convert = (
		tools?.[call.toolName] as
			| {
					toModelOutput?: (options: {
						toolCallId: string;
						input: unknown;
						output: unknown;
					}) => ToolResultPart["output"] | PromiseLike<ToolResultPart["output"]>;
			  }
			| undefined
	)?.toModelOutput;
	if (convert) {
		try {
			return await convert({
				toolCallId: call.toolCallId,
				input: result.input,
				output: result.value,
			});
		} catch {
			// Cancellation durability must not depend on an app converter. Fall
			// through to the SDK's default JSON-shaped representation.
		}
	}
	return { type: "json", value: safeJsonValue(result.value) as never };
}

async function messagesForCancelledStep(
	capture: PartialStepCapture,
	tools: ToolSet | undefined,
): Promise<ModelMessage[]> {
	const assistantContent: Array<{ type: "text"; text: string } | CapturedToolCall> = [];
	if (capture.text.length > 0) assistantContent.push({ type: "text", text: capture.text });
	assistantContent.push(...capture.toolCalls);
	if (assistantContent.length === 0) return [];

	const messages: ModelMessage[] = [{ role: "assistant", content: assistantContent }];
	if (capture.toolCalls.length === 0) return messages;

	const results: ToolResultPart[] = [];
	for (const call of capture.toolCalls) {
		const captured = capture.toolResults.get(call.toolCallId);
		results.push({
			type: "tool-result",
			toolCallId: call.toolCallId,
			toolName: call.toolName,
			output: captured
				? await capturedToolOutput(captured, call, tools)
				: {
						type: "error-text",
						value: "Tool execution was interrupted because the run was cancelled.",
					},
		});
	}
	messages.push({ role: "tool", content: results });
	return messages;
}

function unknownStepUsage(): StepUsage {
	return {
		inputTokens: null,
		outputTokens: null,
		totalTokens: null,
		cachedInputTokens: null,
		reasoningTokens: null,
		cost: null,
		upstreamCost: null,
		isByok: null,
		billedCost: null,
	};
}

const DEFAULT_POKE = (final: boolean) =>
	`You ended your turn without calling submit_deliverable or cancel_task. ` +
	`${final ? "This is your final chance — you MUST" : "You must"} call one of them now: ` +
	`submit_deliverable if the task is done (or can be finished right now), cancel_task if it is impossible.`;

export async function runLoop<TOOLS extends ToolSet>(
	options: RunLoopOptions<TOOLS>,
): Promise<RunResult> {
	const { sessionId, agent, storage } = options;
	const retry = resolveRetryConfig(agent.retry);
	const compaction = resolveCompactionConfig(agent.compaction);
	const maxPokes = options.maxPokes ?? 2;
	const guard = agent.toolResultGuard;
	const tools =
		agent.tools && guard !== false
			? guardToolResultSizes(agent.tools, {
					maxBytes: guard?.maxBytes ?? 120_000,
					mode: guard?.mode ?? "truncate",
				})
			: agent.tools;

	// Every live event is stamped with its emit time here, so construction
	// sites carry only their own payload.
	const emit = (event: EmittableEvent) => {
		try {
			options.emit?.({ ...event, at: now() } as AgenticEvent);
		} catch {
			// a broken listener must never kill a run
		}
	};
	const append = (event: StoredEvent) => storage.append(sessionId, event);

	const runId = options.resumeRunId ?? newId();
	const mailbox = options.mailbox;
	if (mailbox) mailbox.runId = runId;
	let runStarted = options.resumeRunId !== undefined;
	let resultMessageId = options.messageId ?? "";
	let totals: UsageTotals = emptyTotals();
	let finalText = "";
	let attempt = 0;
	let pokes = 0;
	let forceCompact = false;
	let currentInputQueueIds: string[] = [];
	let currentInputMessageIds: string[] = [];
	// Progress since the last forced compaction — if a context-overflow error
	// recurs with no new steps in between, compaction cannot save this run.
	let progressSinceForcedCompaction = true;
	const startedAt = Date.now();

	const fail = async (error: unknown): Promise<RunResult> => {
		if (mailbox) mailbox.accepting = false;
		const serialized = serializeError(error);
		await append({ type: "run-end", at: now(), runId, status: "failed", error: serialized });
		emit({ type: "run-end", sessionId, runId, status: "failed", totals, error: serialized });
		return {
			status: "failed",
			runId,
			messageId: resultMessageId,
			text: finalText,
			totals,
			error: serialized,
		};
	};

	const cancel = async (
		reason: unknown,
		partial?: { messages: ModelMessage[]; text: string; toolCalls: CapturedToolCall[] },
	): Promise<RunResult> => {
		if (mailbox) mailbox.accepting = false;
		const serialized = serializeError(reason ?? "Cancelled");
		if (partial?.text.length) finalText = partial.text;
		if (!runStarted) {
			await append({ type: "run-start", at: now(), runId, model: agent.model });
			runStarted = true;
			emit({ type: "run-start", sessionId, runId, model: agent.model });
		}
		if (runStarted) {
			const usage = unknownStepUsage();
			const stored = storeMessages(
				agent.preserveProviderOptions
					? (partial?.messages ?? [])
					: stripProviderOptions(partial?.messages ?? []),
			);
			try {
				await append({
					type: "step",
					at: now(),
					runId,
					messages: stored,
					inputMessageIds: currentInputMessageIds,
					inputQueueIds: currentInputQueueIds,
					acknowledgesInput: true,
					finishReason: "cancelled",
					usage,
					interrupted: { status: "cancelled", error: serialized },
				});
			} catch (storageError) {
				// Never report a durable cancellation when its visible partial step
				// was not saved. Close as failed when storage still permits it.
				return fail(storageError);
			}
			totals = addStepToTotals(totals, usage);
			emit({
				type: "step",
				sessionId,
				runId,
				finishReason: "cancelled",
				usage,
				messages: stored,
				inputMessageIds: currentInputMessageIds,
				toolCalls: (partial?.toolCalls ?? []).map((call) => ({
					toolName: call.toolName,
					input: call.input,
				})),
				text: partial?.text ?? "",
			});
			currentInputQueueIds = [];
			currentInputMessageIds = [];
		}
		await append({ type: "run-end", at: now(), runId, status: "cancelled", error: serialized });
		emit({ type: "run-end", sessionId, runId, status: "cancelled", totals, error: serialized });
		return {
			status: "cancelled",
			runId,
			messageId: resultMessageId,
			text: finalText,
			totals,
			error: serialized,
		};
	};

	// Summarize-and-rebase the session. "skip" = nothing left to compact away.
	const compactNow = async (): Promise<
		{ outcome: "done" | "skip" } | { outcome: "failed"; error: unknown }
	> => {
		const replayed = replaySession(await storage.load(sessionId));
		// Pending queued inputs must survive compaction verbatim. They can be
		// sandwiched below an in-flight step, so causally hoist them first and
		// expand the recent tail far enough to retain every one.
		const pendingInputCount = replayed.pendingMessages;
		if (pendingInputCount > 0) {
			hoistPendingInputs(replayed.messages, replayed.pendingInputMessages);
		}
		const compactionForPass =
			pendingInputCount > compaction.keepRecent
				? { ...compaction, keepRecent: pendingInputCount }
				: compaction;
		if (replayed.messages.length <= compactionForPass.keepRecent + 1) {
			return { outcome: "skip" };
		}
		try {
			const compacted = await runCompaction({
				messages: replayed.messages.map((entry) => entry.message),
				model: options.getModel(compaction.model ?? agent.model, agent),
				config: compactionForPass,
				abortSignal: options.abortSignal,
			});
			// Identity continuity: a message retained in the base keeps its ledger
			// id and original persist time; only the summary gets a fresh id.
			const identityOf = new Map<ModelMessage, SessionMessage>();
			for (const entry of replayed.messages) identityOf.set(entry.message, entry);
			const persisted = agent.preserveProviderOptions
				? compacted.messages
				: stripProviderOptions(compacted.messages);
			const compactedAt = now();
			const stored: StoredMessage[] = persisted.map((message, index) => {
				const prior = identityOf.get(compacted.messages[index]);
				return {
					id: prior?.id ?? newId(),
					at: prior?.at ? prior.at : compactedAt,
					message,
				};
			});
			await append({
				type: "compaction",
				at: compactedAt,
				messages: stored,
				usage: compacted.usage,
			});
			// cost counts; the summarizer's context size is not the live chat's
			totals = { ...addStepToTotals(totals, compacted.usage), contextTokens: null };
			emit({
				type: "compaction",
				sessionId,
				beforeTokens: replayed.contextTokens,
				summaryChars: compacted.summary.length,
			});
			return { outcome: "done" };
		} catch (err) {
			return { outcome: "failed", error: err };
		}
	};

	while (true) {
		// Queue ids are causal only once a model pass actually starts. In
		// particular, pre-pass compaction can be aborted before the model that
		// answers these inputs is ever called.
		currentInputQueueIds = [];
		currentInputMessageIds = [];
		if (options.abortSignal?.aborted) return cancel(options.abortSignal.reason);

		const events = await storage.load(sessionId);
		const replayed = replaySession(events);
		const inputQueueIds = replayed.pendingQueueIds;
		const inputMessageIds = replayed.pendingInputMessages.map((entry) => entry.id);
		if (resultMessageId === "") resultMessageId = inputMessageIds.at(-1) ?? "";
		if (replayed.pendingMessages > 0) {
			hoistPendingInputs(replayed.messages, replayed.pendingInputMessages);
		}
		if (mailbox && inputQueueIds.length > 0) {
			const included = new Set(inputQueueIds);
			for (const subscription of mailbox.queuedPartListeners) {
				if (included.has(subscription.queueId)) subscription.active = true;
			}
		}
		// Queued live pickup: consume the mailbox arrivals this snapshot has.
		// Anything the load raced past stays queued and triggers another pass.
		if (mailbox && mailbox.queued.length > 0) {
			const inSnapshot = mailbox.queued.filter((id) => ledgerHasQueued(events, id));
			if (inSnapshot.length > 0) {
				mailbox.queued = mailbox.queued.filter((id) => !inSnapshot.includes(id));
			}
		}
		const contextWindow = await getContextWindow(agent.model);
		totals = { ...totals, contextWindow };

		// ── compaction before the pass (threshold crossed, or forced by an
		// overflow error) ──
		const wantCompaction =
			forceCompact || shouldCompact(replayed.contextTokens, contextWindow, compaction.limit);
		if (wantCompaction) {
			const compacted = await compactNow();
			if (compacted.outcome === "done") {
				forceCompact = false;
				progressSinceForcedCompaction = false;
				continue; // replay the compacted session and keep working
			}
			if (compacted.outcome === "failed") {
				if (options.abortSignal?.aborted) return cancel(options.abortSignal.reason);
				const cls = classifyFailure(compacted.error);
				if (cls.kind === "transient" && attempt < retry.maxAttempts) {
					attempt += 1;
					const delayMs = retryDelayMs(attempt, retry, cls.retryAfterMs);
					emit({
						type: "retry",
						sessionId,
						runId,
						attempt,
						maxAttempts: retry.maxAttempts,
						delayMs,
						error: cls.error,
					});
					await wait(delayMs, options.abortSignal).catch(() => {});
					continue;
				}
				return fail(compacted.error);
			}
			// outcome "skip": nothing left to compact away
			if (forceCompact) {
				// The provider says the conversation is too big and there is
				// nothing left to compact — deterministic, unrecoverable.
				return fail(
					new Error(
						"Context window overflow that compaction cannot fix (conversation is already minimal)",
					),
				);
			}
		}

		if (!runStarted) {
			runStarted = true;
			await append({ type: "run-start", at: now(), runId, model: agent.model });
			emit({ type: "run-start", sessionId, runId, model: agent.model });
		}

		// The transcript identity this pass streams into. projectRun over the
		// pass snapshot yields the segments prior steps created; this pass's step
		// begins a new segment only when it folds in input no existing segment
		// carries while the open segment already has messages — the exact
		// beginOrExtend condition projection applies when the step is later
		// replayed, so the id can never drift from projectSession's. Constant
		// across retries of the same step (the snapshot and pending inputs are
		// unchanged until a step durably lands).
		const priorSegments = projectRun(events, runId)?.segments ?? [];
		const claimedInputIds = new Set(
			priorSegments.flatMap((segment) => segment.inputs.map((input) => input.id)),
		);
		const lastSegment = priorSegments.at(-1);
		const beginsSegment =
			lastSegment !== undefined &&
			lastSegment.messages.length > 0 &&
			inputMessageIds.some((inputId) => !claimedInputIds.has(inputId));
		const segmentIndex = beginsSegment
			? priorSegments.length
			: Math.max(priorSegments.length - 1, 0);
		const streamContext: StreamContext = { runId, responseId: `${runId}/${segmentIndex}` };

		// ── one streamText pass ──
		let compactPending = false;
		let progressThisPass = false;
		let observableProgress = false;
		let lastFinishReason = "";
		let streamError: unknown;
		let queueIdsForNextStep = inputQueueIds;
		let messageIdsForNextStep = inputMessageIds;
		const partialStep = emptyPartialStepCapture();
		// onStepFinish persistence is async; every pending append must settle
		// before the next pass replays the session, or the replay races the
		// writes it depends on.
		const pendingAppends: Promise<unknown>[] = [];

		// Storage/context lookup, compaction, and run-start persistence above can
		// all yield. Do not mark queued inputs causal if cancellation landed
		// before streamText is actually entered.
		if (options.abortSignal?.aborted) return cancel(options.abortSignal.reason);
		currentInputQueueIds = inputQueueIds;
		currentInputMessageIds = inputMessageIds;
		const result = streamText({
			model: options.getModel(agent.model, agent),
			system: agent.system,
			messages: replayed.messages.map((entry) => entry.message),
			tools,
			toolChoice: agent.toolChoice,
			abortSignal: options.abortSignal,
			// The framework owns ALL retry policy — a second retry layer inside
			// the SDK would hide failures from the classifier and double waits.
			maxRetries: 0,
			// Errors are handled via the error chunk below; without this the SDK
			// dumps them to console.
			onError: () => {},
			stopWhen: [
				stepCountIs(agent.maxSteps ?? 50),
				() => compactPending || options.isSettled?.() === true,
				// a queued message arrived mid-pass: stop at this step boundary so
				// the next pass folds it into the conversation
				() => (mailbox?.queued.length ?? 0) > 0,
				...(agent.stopWhen ?? []),
			],
			onStepFinish: (step) => {
				// This step is now represented by step.response.messages. Clear the
				// live-part salvage buffer synchronously so an abort from an event
				// listener cannot persist the same assistant output twice.
				resetPartialStepCapture(partialStep);
				const usage: StepUsage = extractStepUsage({
					usage: step.usage,
					providerMetadata: step.providerMetadata as Record<string, unknown> | undefined,
				});
				const newMessages = step.response.messages.slice();
				const acknowledgesInput = step.finishReason !== "error" && newMessages.length > 0;
				const stepInputQueueIds = queueIdsForNextStep;
				const stepInputMessageIds = messageIdsForNextStep;
				if (acknowledgesInput) {
					queueIdsForNextStep = [];
					messageIdsForNextStep = [];
					currentInputQueueIds = [];
					currentInputMessageIds = [];
				}
				const stored = storeMessages(
					agent.preserveProviderOptions ? newMessages : stripProviderOptions(newMessages),
				);
				pendingAppends.push(
					Promise.resolve(
						append({
							type: "step",
							at: now(),
							runId,
							messages: stored,
							inputMessageIds: stepInputMessageIds,
							inputQueueIds: stepInputQueueIds,
							acknowledgesInput,
							finishReason: step.finishReason,
							usage,
						}),
					),
				);
				totals = addStepToTotals(totals, usage);
				lastFinishReason = step.finishReason;
				if (step.text.trim()) finalText = step.text;
				if (step.toolCalls.length > 0 || step.text.trim() || (usage.outputTokens ?? 0) > 0) {
					progressThisPass = true;
					progressSinceForcedCompaction = true;
				}
				emit({
					type: "step",
					sessionId,
					runId,
					finishReason: step.finishReason,
					usage,
					messages: stored,
					inputMessageIds: stepInputMessageIds,
					toolCalls: step.toolCalls.map((t) => ({ toolName: t.toolName, input: t.input })),
					text: step.text,
				});
				if (shouldCompact(contextTokensOf(usage), contextWindow, compaction.limit)) {
					compactPending = true;
				}
			},
		});

		// Drain the stream, capturing the last provider error chunk. The SDK
		// surfaces mid-stream provider drops as an `error` chunk + finishReason
		// "error" rather than throwing — and can also throw at flush (e.g.
		// AI_NoOutputGeneratedError). Keep the richer chunk when both happen.
		try {
			for await (const part of result.fullStream) {
				captureStreamPart(partialStep, part);
				if (part.type === "error") streamError = part.error;
				if (
					part.type === "text-delta" ||
					part.type === "tool-call" ||
					part.type === "tool-result"
				) {
					observableProgress = true;
				}
				try {
					options.onPart?.(part, streamContext);
				} catch {
					// a broken part listener must never kill a run
				}
				for (const listener of mailbox?.partListeners ?? []) {
					try {
						listener(part, streamContext);
					} catch {
						// same rule for attached listeners
					}
				}
				for (const subscription of mailbox?.queuedPartListeners ?? []) {
					if (!subscription.active) continue;
					try {
						subscription.listener(part, streamContext);
					} catch {
						// same rule for causally attached listeners
					}
				}
			}
		} catch (err) {
			if (streamError === undefined) streamError = err;
		}
		try {
			await Promise.all(pendingAppends);
		} catch (storageError) {
			// A step that was not durably recorded cannot be replayed after a
			// restart. Treat that as a run failure instead of claiming completion
			// over state that exists only in this process.
			return fail(storageError);
		}

		if (options.abortSignal?.aborted) {
			return cancel(options.abortSignal.reason, {
				messages: await messagesForCancelledStep(partialStep, tools),
				text: partialStep.text,
				toolCalls: partialStep.toolCalls,
			});
		}

		// ── pass ended with a provider/stream error ──
		if (streamError !== undefined) {
			const cls = classifyFailure(streamError);

			if (cls.kind === "context-overflow") {
				// Compaction replays the prompt. Once callers saw text or a tool ran,
				// replaying could duplicate visible output or external side effects.
				if (observableProgress) return fail(streamError);
				if (!progressSinceForcedCompaction) {
					return fail(streamError); // compaction already tried and didn't help
				}
				forceCompact = true;
				continue;
			}

			if (cls.kind === "fatal") return fail(streamError);
			// Retrying after callers saw text or a tool executed would duplicate
			// visible output or side effects against the same prompt.
			if (observableProgress) return fail(streamError);

			// transient — bounded by attempts-without-progress and wall clock
			if (progressThisPass) attempt = 0;
			attempt += 1;
			if (attempt > retry.maxAttempts) return fail(streamError);
			if (Date.now() - startedAt > retry.maxElapsedMs) return fail(streamError);
			const delayMs = retryDelayMs(attempt, retry, cls.retryAfterMs);
			emit({
				type: "retry",
				sessionId,
				runId,
				attempt,
				maxAttempts: retry.maxAttempts,
				delayMs,
				error: cls.error,
			});
			try {
				await wait(delayMs, options.abortSignal);
			} catch (abortReason) {
				return cancel(abortReason);
			}
			continue;
		}

		// ── pass ended cleanly ──
		// The pass was stopped mid-work for compaction (last step still had
		// tool calls pending) — compact at the top of the next pass, then the
		// model picks its task back up.
		if (compactPending && lastFinishReason === "tool-calls") continue;

		// Unconsumed queued messages: the run does not end while there is
		// unanswered user input (opencode's exit rule) — unless the task has
		// already settled, in which case the outcome stands.
		if (mailbox && mailbox.queued.length > 0 && options.isSettled?.() !== true) {
			// A signal can arrive just after the loop independently discovered and
			// answered its ledger message. Re-check causal pending state so that
			// stale signal does not force a duplicate model pass.
			const latestEvents = await storage.load(sessionId);
			const latestPending = new Set(replaySession(latestEvents).pendingQueueIds);
			mailbox.queued = mailbox.queued.filter((id) => latestPending.has(id));
			if (mailbox.queued.length > 0) continue;
		}

		if (options.isSettled && !options.isSettled()) {
			pokes += 1;
			if (pokes > maxPokes) {
				return fail(
					new Error(
						`Run ended without settling the task after ${maxPokes} pokes (model never called a terminal tool)`,
					),
				);
			}
			const content = options.pokeMessage?.(pokes) ?? DEFAULT_POKE(pokes === maxPokes);
			const pokeMessageId = newId();
			await append({
				type: "user-message",
				at: now(),
				id: pokeMessageId,
				message: { role: "user", content },
				meta: { poke: pokes },
			});
			emit({ type: "poke", sessionId, runId, poke: pokes, maxPokes, messageId: pokeMessageId });
			continue;
		}

		// Point of no return: refuse new mailbox enqueues synchronously BEFORE
		// the run-end append, so a send() racing this exit falls back to
		// starting its own run instead of attaching to one that won't look.
		if (mailbox) mailbox.accepting = false;
		await append({ type: "run-end", at: now(), runId, status: "completed" });
		emit({ type: "run-end", sessionId, runId, status: "completed", totals });
		// The model finished its turn over the threshold — compact silently
		// AFTER the run so the next turn starts fresh, instead of re-entering
		// the model just to regenerate an answer it already gave. A failure
		// here is non-fatal: the threshold is still crossed, so the next run
		// retries it.
		if (compactPending) await compactNow();
		return { status: "completed", runId, messageId: resultMessageId, text: finalText, totals };
	}
}
