import type { LanguageModel, ModelMessage, TextStreamPart, ToolSet } from "ai";
import { tool } from "ai";
import { type ZodType, z } from "zod";
import { resolveRetryConfig, retryDelayMs, wait } from "./backoff.js";
import { classifyFailure, serializeError } from "./failure.js";
import { logEvents } from "./logEvents.js";
import { getContextWindow } from "./modelMeta.js";
import { createOpenRouter } from "./openrouter.js";
import type { ProgressEvent } from "./progress.js";
import {
	projectSession,
	type SessionTranscript,
	type StreamContext,
	textOf,
} from "./projection.js";
import { type ReplayedSession, replaySession } from "./replay.js";
import { createResilientFetch, type ResilientFetchOptions } from "./resilientFetch.js";
import { createMailbox, type RunDelivery, type RunLoopOptions, runLoop } from "./run.js";
import { fileStorage, memoryStorage, serializedStorage } from "./storage.js";
import type {
	AgentConfig,
	EventListener,
	MaybePromise,
	RunResult,
	SessionMessage,
	StorageProvider,
	StoredEvent,
	TaskOutcome,
	UsageTotals,
} from "./types.js";

/**
 * Maps a session id to the agent config that should resume it, or null/
 * undefined to leave that session alone. Configs hold live tool functions,
 * so they can't be persisted — this is how the app re-supplies them.
 */
export type AutoResumeResolver = (
	sessionId: string,
) => MaybePromise<AgentConfig<ToolSet> | null | undefined>;

export interface AutoResumeOptions {
	/** Re-supply the agent config for a session found interrupted. */
	agentFor: AutoResumeResolver;
	/**
	 * Crash-loop breaker: after this many automatic resume attempts on the
	 * same interrupted work (counted in the ledger, so it survives restarts),
	 * the sweep stops retrying and emits an auto-resume give-up event.
	 * Manual resume() calls are never capped. Default 3.
	 */
	maxAttempts?: number;
	/**
	 * Maximum auto-resumed runs in flight for this runtime. Must be a positive
	 * integer. Default 4.
	 */
	maxConcurrent?: number;
	/** Pause between resume kicks so a big boot doesn't stampede the provider. Default 0. */
	staggerMs?: number;
}

export interface AgenticOptions {
	/** Defaults to OPENROUTER_API_KEY from the environment. */
	apiKey?: string;
	/**
	 * Where sessions live. Defaults to JSONL files under ./.agentic. Use one
	 * createAgentic instance per process for a given provider: locks and live
	 * mailboxes are instance-local.
	 */
	storage?: StorageProvider;
	/**
	 * Make crash recovery automatic: as soon as the harness is created it
	 * sweeps storage for interrupted work (open runs, unanswered queued
	 * messages) and resumes each session in the background, using this
	 * resolver to map session ids back to agent configs. Pass the resolver
	 * directly or wrap it in {@link AutoResumeOptions} to tune the attempt cap,
	 * concurrency, and stagger. resumeInterrupted() runs the same sweep on
	 * demand.
	 */
	autoResume?: AutoResumeResolver | AutoResumeOptions;
	/** Observability firehose: run/step/retry/compaction/poke events. */
	onEvent?: EventListener;
	/**
	 * Log every event to the console, one colored line each (see
	 * {@link logEvents}). Composes with onEvent — the log line prints first.
	 */
	logs?: boolean;
	/** App attribution etc., sent on every request. */
	headers?: Record<string, string>;
	/** Provider routing etc., merged into every request body. */
	extraBody?: Record<string, unknown>;
	/** Header/SSE-idle stall detection. Defaults 60s/120s; false disables. */
	fetchTimeouts?: ResilientFetchOptions | false;
	/**
	 * Advanced: override model construction (custom providers, tests). When
	 * set, the OpenRouter factory is bypassed for model resolution.
	 */
	// biome-ignore lint/suspicious/noExplicitAny: contravariant position — any AgentConfig works
	getModel?: (modelId: string, agent: AgentConfig<any>) => LanguageModel;
}

export interface SendOptions<TOOLS extends ToolSet = ToolSet> {
	/**
	 * Best-effort raw stream tap for the initiating caller: every AI SDK
	 * stream part with its {@link StreamContext}. Parts are not replayed from
	 * the ledger; the resolved RunResult/session.messages() remain the durable
	 * source of truth. Most apps want {@link onProgress} instead — reach for
	 * onPart only when you need the raw parts themselves.
	 */
	onPart?: (part: TextStreamPart<TOOLS>, context: StreamContext) => void;
	/**
	 * Canonical live progress for this send: {@link ProgressEvent}s
	 * (response-start, text/reasoning deltas, tool activity, retry, run-end) —
	 * JSON-serializable, forwardable over SSE/WebSocket verbatim. Live-only
	 * and best-effort like onPart; a throwing listener is contained and never
	 * breaks a run.
	 */
	onProgress?: (event: ProgressEvent) => void;
	/**
	 * Send-time identities: fires exactly once per send, after the user-message
	 * event is durably appended and a run is registered/joined for it, and
	 * always before this send's first onPart delivery. For a queued send the
	 * ultimately-answering run can differ (successor collapse) — RunResult.runId
	 * remains the final authority, and UIs should key rows on transcript ids,
	 * which never lie. A throwing callback is contained (like onPart) and never
	 * corrupts the run.
	 */
	onAccepted?: (info: {
		/** The durable user-message ledger id for THIS send. */
		messageId: string;
		/** Fresh send: the run created for it. Queued send: the live run it joined. */
		runId: string;
		queued: boolean;
		/** The user-message event's timestamp. */
		at: string;
	}) => void;
	abortSignal?: AbortSignal;
	/** App-supplied tag stored on the user-message event (dedup keys etc.). */
	meta?: Record<string, unknown>;
	/**
	 * When the session has a live run, queue this message into it: the message
	 * is appended to the ledger first (durable — a crash cannot drop it), the
	 * run folds it in at its next step boundary, and this send resolves with
	 * that run's result. Set false to wait for the live run to finish and get
	 * a dedicated run instead. Default true.
	 */
	queue?: boolean;
}

export interface Session<TOOLS extends ToolSet = ToolSet> {
	id: string;
	/**
	 * Append a user message and run the agent to completion. If a run is
	 * already live, the message is queued into it (see SendOptions.queue).
	 */
	send(content: string | ModelMessage, options?: SendOptions<TOOLS>): Promise<RunResult>;
	/**
	 * Re-enter work a crash/restart left behind: an open run (run-start with
	 * no run-end) or queued user messages that never got a response. Resolves
	 * null when there is nothing to resume.
	 */
	resume(
		options?: Pick<SendOptions<TOOLS>, "onPart" | "onProgress" | "abortSignal">,
	): Promise<RunResult | null>;
	/**
	 * Session-scoped live tail — the transport side of progress: while
	 * attached, the listener receives every {@link ProgressEvent} from ANY run
	 * executing for this session in this process — the currently live run and
	 * runs that start later, tasks included. Live-only and best-effort like
	 * onProgress: nothing is replayed, and a throwing listener is contained
	 * and never breaks a run. Attaching while idle is valid and simply
	 * delivers nothing until a run starts. Returns a detach function. (Raw
	 * stream parts are the initiating caller's tap: SendOptions.onPart.)
	 *
	 * Reconnect recipe: attach first (buffering events), then read
	 * transcript(); drop buffered/incoming `text` events whose `offset` is
	 * less than the streaming item's `partialText.length` and append the rest
	 * — the snapshot and the stream can then never double-render a character.
	 * On every `response-start`, reset the expected offset to 0 and (re)key
	 * the bubble by its `responseId`: it marks a new model pass, whose deltas
	 * start over at offset 0.
	 */
	attach(listener: (event: ProgressEvent) => void): () => void;
	/**
	 * The replay-ready conversation as the model would see it, each message
	 * carrying its durable ledger id and persist time.
	 */
	messages(): Promise<SessionMessage[]>;
	/**
	 * The renderable conversation: user turns and response segments in causal
	 * display order with per-item lifecycle status (see
	 * {@link projectSession}). Live-ness is wired from this runtime's
	 * in-process run registry — the same source isRunning() uses — so an open
	 * run projects as "streaming" here and as "interrupted" after a crash.
	 * `internal: true` includes framework-internal messages (pokes).
	 */
	transcript(options?: { internal?: boolean }): Promise<SessionTranscript>;
	/** Lifetime usage/cost totals plus current context pressure. */
	stats(): Promise<UsageTotals>;
	/** True when a run was interrupted and resume() would do work. */
	isInterrupted(): Promise<boolean>;
}

export interface TaskOptions<T, TOOLS extends ToolSet = ToolSet> {
	agent: AgentConfig<TOOLS>;
	/** The task brief, appended as the first user message. */
	prompt: string | ModelMessage[];
	/**
	 * Validated shape of the deliverable. Validation errors are returned TO
	 * THE MODEL as tool results so it can fix them — no thrown parse errors,
	 * no memory-less retries. Omit for free-text deliverables.
	 */
	deliverable?: ZodType<T>;
	/** Session id — pass one to make the task resumable/auditable by name. */
	id?: string;
	/**
	 * Persist this task in the configured ledger. Set false for presentation
	 * helpers and other one-shot work that should not create a recoverable
	 * session. Defaults true.
	 */
	durable?: boolean;
	/** Extra instructions appended to the submit tool's description. */
	deliverableHint?: string;
	maxPokes?: number;
	abortSignal?: AbortSignal;
	onPart?: (part: TextStreamPart<TOOLS>, context: StreamContext) => void;
	/** Canonical live progress, like {@link SendOptions.onProgress}. */
	onProgress?: (event: ProgressEvent) => void;
}

export interface Agentic {
	session<TOOLS extends ToolSet>(id: string, agent: AgentConfig<TOOLS>): Session<TOOLS>;
	/**
	 * Run a workflow task with a guaranteed outcome: the model MUST finish by
	 * calling submit_deliverable (validated) or cancel_task (its escape hatch
	 * for impossible/malformed/unsafe asks). Ending a turn without either
	 * gets it poked, up to maxPokes. Never throws for model behavior — the
	 * outcome status says what happened.
	 */
	task<T = string, TOOLS extends ToolSet = ToolSet>(
		options: TaskOptions<T, TOOLS>,
	): Promise<TaskOutcome<T>>;
	/** Session ids known to storage (when the provider supports listing). */
	sessions(): Promise<string[]>;
	/**
	 * Config-free read of a session's renderable transcript — identical to
	 * `session(id, agent).transcript()` (same projection, same live overlay)
	 * but with no AgentConfig involved: reading never runs the model, so
	 * read-only routes don't build tools/prompts just to render a chat. Pair
	 * with {@link wireTranscript} for the JSON-safe client payload.
	 */
	transcript(sessionId: string, opts?: { internal?: boolean }): Promise<SessionTranscript>;
	/**
	 * Sessions with recoverable work — an interrupted run or queued user
	 * messages that never got a response. Runs live in this process don't
	 * count. With autoResume configured these are swept up automatically;
	 * otherwise feed them to resume() on boot.
	 */
	interruptedSessions(): Promise<string[]>;
	/**
	 * Run the auto-resume sweep now (it already runs once, in the background,
	 * when the harness is created): resume every interrupted session whose
	 * config the autoResume resolver supplies, skipping any that exhausted
	 * the attempt cap. Resolves with the finished runs' results. Throws if
	 * autoResume was not configured.
	 */
	resumeInterrupted(): Promise<RunResult[]>;
	/** True while this runtime has an in-process run registered for the session. */
	isRunning(sessionId: string): boolean;
	/**
	 * Explicitly cancel an in-process run, including an automatically resumed
	 * run. Returns true only when this call newly requested cancellation.
	 */
	cancel(sessionId: string, reason?: unknown): boolean;
}

function toUserMessage(content: string | ModelMessage): ModelMessage {
	return typeof content === "string" ? { role: "user", content } : content;
}

function newId(): string {
	return globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2);
}

export function createAgentic(options: AgenticOptions = {}): Agentic {
	const storage = serializedStorage(options.storage ?? fileStorage());
	const openrouter = createOpenRouter({
		apiKey: options.apiKey,
		headers: options.headers,
		extraBody: options.extraBody,
		fetch:
			options.fetchTimeouts === false
				? undefined
				: createResilientFetch(options.fetchTimeouts ?? {}),
	});

	// biome-ignore lint/suspicious/noExplicitAny: only extraBody is read
	const getModel = (modelId: string, agent: AgentConfig<any>): LanguageModel =>
		options.getModel
			? options.getModel(modelId, agent)
			: openrouter.chat(modelId, agent.extraBody ? { extraBody: agent.extraBody } : {});

	// One run at a time per session: concurrent send()s queue up rather than
	// interleave steps into the same ledger.
	const locks = new Map<string, Promise<unknown>>();
	function withSessionLock<T>(sessionId: string, fn: () => Promise<T>): Promise<T> {
		const prev = locks.get(sessionId) ?? Promise.resolve();
		const next = prev.then(fn, fn);
		const tail = next.then(
			() => undefined,
			() => undefined,
		);
		locks.set(sessionId, tail);
		void tail.then(() => {
			if (locks.get(sessionId) === tail) locks.delete(sessionId);
		});
		return next;
	}

	// The session's live run, if any — what send() queues messages into.
	interface LiveRun {
		mailbox: ReturnType<typeof createMailbox>;
		result: Promise<RunResult>;
		controller: AbortController;
	}
	const liveRuns = new Map<string, LiveRun>();

	// Session-scoped live-tail listeners (Session.attach). Keyed by session id
	// and read at delivery time, so attaching/detaching mid-run takes effect
	// immediately and one registration spans every future run.
	const attachedListeners = new Map<string, Set<(event: ProgressEvent) => void>>();

	// Start a run registered in the live-run map so concurrent send()s can
	// queue into it. Must be called while holding the session lock.
	function execRun<TOOLS extends ToolSet>(
		sessionId: string,
		agent: AgentConfig<TOOLS>,
		runOptions: Partial<
			Pick<
				RunLoopOptions<TOOLS>,
				| "abortSignal"
				| "onPart"
				| "onProgress"
				| "resumeRunId"
				| "isSettled"
				| "pokeMessage"
				| "maxPokes"
				| "messageId"
			>
		>,
		mailbox = createMailbox(),
		controller = new AbortController(),
		runStorage: StorageProvider = storage,
	): Promise<RunResult> {
		const initiatingSignal = runOptions.abortSignal;
		let detachAbortListener: (() => void) | undefined;
		if (initiatingSignal) {
			const forwardAbort = () => {
				// Once another durable caller joins, the initiating HTTP/client
				// disconnect no longer owns the shared run. Its queued work must finish.
				if (mailbox.attachedQueueIds.size === 0) {
					controller.abort(initiatingSignal.reason);
				}
			};
			if (initiatingSignal.aborted) forwardAbort();
			else {
				initiatingSignal.addEventListener("abort", forwardAbort, { once: true });
				detachAbortListener = () => initiatingSignal.removeEventListener("abort", forwardAbort);
			}
		}
		// Every run entry point flows through here, so composing the session's
		// attach() broadcast over the caller's onProgress makes live-tailing
		// cover fresh sends, queued recovery, resume, the auto-resume sweep,
		// and tasks alike. The registry is read per event, so mid-run
		// attach/detach works. Raw parts stay the initiating caller's private
		// tap (onPart) — attach is the transport tail and speaks ProgressEvent.
		const callerOnProgress = runOptions.onProgress;
		const onProgress = (event: ProgressEvent) => {
			if (callerOnProgress) {
				try {
					callerOnProgress(event);
				} catch {
					// a broken caller listener must not starve attached listeners
				}
			}
			const attached = attachedListeners.get(sessionId);
			if (!attached || attached.size === 0) return;
			for (const listener of [...attached]) {
				try {
					listener(event);
				} catch {
					// error-contained, like onProgress
				}
			}
		};
		const result = runLoop<TOOLS>({
			sessionId,
			agent,
			storage: runStorage,
			getModel,
			emit: emitEvent,
			mailbox,
			...runOptions,
			onProgress,
			abortSignal: controller.signal,
		});
		liveRuns.set(sessionId, { mailbox, result, controller });
		// runLoop assigns mailbox.runId synchronously before its first await, so
		// acceptance hooks registered on this mailbox observe the real run id
		// (including a resumed run's) before any stream part can be delivered.
		if (mailbox.runId !== null) {
			const startedRunId = mailbox.runId;
			for (const hook of mailbox.acceptHooks.splice(0)) hook(startedRunId);
		}
		const cleanup = () => {
			detachAbortListener?.();
			if (liveRuns.get(sessionId)?.mailbox === mailbox) liveRuns.delete(sessionId);
		};
		result.then(cleanup, cleanup);
		return result;
	}

	const emitEvent: EventListener = (event) => {
		try {
			if (options.logs) logEvents(event);
			options.onEvent?.(event);
		} catch {
			// a broken listener must never break a send
		}
	};

	// The one transcript read path — Session.transcript() and the config-free
	// Agentic.transcript() are the same projection over the same live overlay.
	async function transcriptOf(
		sessionId: string,
		transcriptOptions: { internal?: boolean } = {},
	): Promise<SessionTranscript> {
		const transcript = projectSession(await storage.load(sessionId), {
			live: liveRuns.has(sessionId),
			internal: transcriptOptions.internal,
		});
		// Overlay the in-flight pass's text-so-far onto its streaming response
		// item (see TranscriptResponseItem.partialText). projectSession stays
		// pure over the ledger; the mirror lives on the live run's mailbox and
		// is nulled the moment the pass's step persists or the run ends.
		const partial = liveRuns.get(sessionId)?.mailbox.partial;
		if (partial) {
			for (const item of transcript.items) {
				if (
					item.kind === "response" &&
					item.id === partial.responseId &&
					item.status === "streaming"
				) {
					item.partialText = partial.text;
				}
			}
		}
		return transcript;
	}

	async function finalizePersistedRun<TOOLS extends ToolSet>(
		sessionId: string,
		agent: AgentConfig<TOOLS>,
		events: StoredEvent[],
		replayed: ReplayedSession,
		runStorage: StorageProvider = storage,
	): Promise<RunResult | null> {
		const runId = replayed.interruptedRunId;
		if (!runId) return null;
		const interrupted = interruptedStepForRun(events, runId);
		if (!interrupted && (replayed.pendingMessages > 0 || !finalizableStep(events, runId))) {
			return null;
		}
		const status = interrupted?.interrupted?.status ?? "completed";
		const error = interrupted?.interrupted?.error;

		await runStorage.append(sessionId, {
			type: "run-end",
			at: new Date().toISOString(),
			runId,
			status,
			preservePending: true,
			...(error ? { error } : {}),
		});
		const totals = {
			...replayed.totals,
			contextWindow: await getContextWindow(agent.model),
		};
		emitEvent({
			type: "run-end",
			at: new Date().toISOString(),
			sessionId,
			runId,
			status,
			totals,
			...(error ? { error } : {}),
		});
		return {
			status,
			runId,
			messageId: latestInputMessageId(events, replayed),
			text: lastAssistantTextForRun(events, runId),
			totals,
			...(error ? { error } : {}),
		};
	}

	function session<TOOLS extends ToolSet>(id: string, agent: AgentConfig<TOOLS>): Session<TOOLS> {
		async function finishQueuedSend(
			queueId: string,
			live: LiveRun,
			sendOptions: SendOptions<TOOLS>,
			acceptQueued: (runId: string) => void,
		): Promise<RunResult> {
			let liveResult: RunResult | undefined;
			let liveFailure: unknown;
			try {
				liveResult = await live.result;
			} catch (error) {
				liveFailure = error;
			}

			const result = await withSessionLock(id, async () => {
				let events = await storage.load(id);
				let replayed = replaySession(events);
				// A cancellation step is terminal even if its run-end append failed.
				// Close it before deciding whether causally-later queued input needs
				// a fresh recovery run.
				const reconciled = await finalizePersistedRun(id, agent, events, replayed);
				if (reconciled) {
					events = await storage.load(id);
					replayed = replaySession(events);
				}
				if (replayed.pendingQueueIds.includes(queueId)) {
					// The prior run ended without any persisted model step whose input
					// contained this accepted message. Re-drive it now; the session lock
					// makes concurrent queued callers collapse into one recovery run.
					const resumeRunId = replayed.interruptedRunId ?? undefined;
					if (resumeRunId) {
						// A failed run-end write leaves the original run open. Continue and
						// close that same run instead of creating a successor while the old
						// id remains permanently interrupted.
						await storage.append(id, {
							type: "run-resume",
							at: new Date().toISOString(),
							runId: resumeRunId,
						});
					}
					const pendingQueueIds = new Set(replayed.pendingQueueIds);
					const recoveryMailbox = createMailbox();
					// Attached callers' live listeners follow their still-pending input
					// into the recovery run. Gating resets: the recovery pass that folds
					// the input in re-activates each one (its first pass replays every
					// pending input, so nothing stays dark).
					recoveryMailbox.queuedListeners.push(
						...live.mailbox.queuedListeners
							.filter((subscription) => pendingQueueIds.has(subscription.queueId))
							.map((subscription) => ({ ...subscription, active: false })),
					);
					// The lock winner may disconnect, but it does not own a recovery
					// shared with other already-durable callers. Seed their identities
					// before execRun wires the initiating abort signal.
					for (const pendingId of pendingQueueIds) {
						if (pendingId !== queueId) recoveryMailbox.attachedQueueIds.add(pendingId);
					}
					// Sends whose acceptance never observed a run id (the joined run
					// died before starting) are accepted by the recovery run instead.
					recoveryMailbox.acceptHooks.push(...live.mailbox.acceptHooks.splice(0));
					return execRun<TOOLS>(
						id,
						agent,
						{
							abortSignal: sendOptions.abortSignal,
							resumeRunId,
							messageId: queueId,
						},
						recoveryMailbox,
					);
				}
				if (reconciled && queueAnsweredByRun(events, queueId) === live.mailbox.runId) {
					return reconciled;
				}

				const answeredRunId = queueAnsweredByRun(events, queueId);
				if (answeredRunId === null || answeredRunId === live.mailbox.runId) {
					if (liveResult) return liveResult;
					if (liveFailure !== undefined) {
						// The model may have durably answered this queue item and only
						// failed while appending run-end. Reconcile that final step before
						// surfacing the storage failure to an already-answered caller.
						if (
							answeredRunId === live.mailbox.runId &&
							replayed.interruptedRunId === answeredRunId
						) {
							const finalized = await finalizePersistedRun(id, agent, events, replayed);
							if (finalized) return finalized;
						}
						throw liveFailure;
					}
				}

				if (answeredRunId !== null) {
					const terminal = terminalEventForRun(events, answeredRunId);
					if (terminal) {
						// Another queued caller recovered a batch containing this message
						// while we waited for the lock. Mirror that run's durable terminal
						// result; a tool-call step followed by failure is not completion.
						return {
							status: terminal.status,
							runId: answeredRunId,
							messageId: queueId,
							text: lastAssistantTextForRun(events, answeredRunId),
							totals: {
								...replayed.totals,
								contextWindow: await getContextWindow(agent.model),
							},
							error: terminal.error,
						};
					}

					if (replayed.interruptedRunId === answeredRunId) {
						// A crash/storage failure can leave the causally answering run open.
						// Reconcile a clean final step, otherwise continue from its durable
						// messages (not from the original run that missed this queue item).
						const finalized = await finalizePersistedRun(id, agent, events, replayed);
						if (finalized) return finalized;
						acceptQueued(answeredRunId);
						return execRun<TOOLS>(id, agent, {
							abortSignal: sendOptions.abortSignal,
							onPart: sendOptions.onPart,
							onProgress: sendOptions.onProgress,
							resumeRunId: answeredRunId,
							messageId: queueId,
						});
					}
				}

				throw new Error(
					`Queued input ${queueId} was recorded as answered without a durable terminal run`,
				);
			});
			// Degenerate races (the joined run never started and an unrelated run
			// answered from the ledger) settle acceptance with the authoritative
			// answering run; the guarded callback makes this a no-op otherwise.
			acceptQueued(result.runId);
			// A merged result reports the answering run — but always THIS send's
			// own durable message id.
			return { ...result, messageId: queueId };
		}

		return {
			id,
			send(content, sendOptions = {}) {
				const message = toUserMessage(content);
				const live = sendOptions.queue !== false ? liveRuns.get(id) : undefined;
				if (live) {
					const queueId = newId();
					live.mailbox.attachedQueueIds.add(queueId);
					return (async () => {
						const at = new Date().toISOString();
						// Persist FIRST — the message survives even if everything
						// after this line dies. Then signal the live run.
						try {
							await storage.append(id, {
								type: "user-message",
								at,
								message,
								id: queueId,
								meta: { ...sendOptions.meta, queued: true, queueId },
							});
						} catch (error) {
							live.mailbox.attachedQueueIds.delete(queueId);
							throw error;
						}
						let acceptedOnce = false;
						const acceptQueued = (runId: string) => {
							if (acceptedOnce) return;
							acceptedOnce = true;
							try {
								sendOptions.onAccepted?.({ messageId: queueId, runId, queued: true, at });
							} catch {
								// contained, like onPart: a broken listener must never break a send
							}
						};
						// The joined run's id, when it is already executing; a run that is
						// registered but not yet started flushes the hook before any part.
						if (live.mailbox.runId !== null) acceptQueued(live.mailbox.runId);
						else live.mailbox.acceptHooks.push(acceptQueued);
						if (sendOptions.onPart || sendOptions.onProgress) {
							// One union listener carries both taps down the shared channel;
							// error containment lives at the delivery site in the loop.
							const { onPart, onProgress } = sendOptions;
							live.mailbox.queuedListeners.push({
								queueId,
								listener: (delivery: RunDelivery) => {
									if (delivery.kind === "part") {
										onPart?.(delivery.part as TextStreamPart<TOOLS>, delivery.context);
									} else {
										onProgress?.(delivery.event);
									}
								},
								active: false,
							});
						}
						if (live.mailbox.tryEnqueue(queueId)) {
							emitEvent({
								type: "queued-message",
								at: new Date().toISOString(),
								sessionId: id,
								runId: live.mailbox.runId,
								messageId: queueId,
							});
							return finishQueuedSend(queueId, live, sendOptions, acceptQueued);
						}
						// The run committed to ending while the append settled. Await it,
						// then causally verify whether it saw this durable message.
						return finishQueuedSend(queueId, live, sendOptions, acceptQueued);
					})();
				}
				if (sendOptions.queue !== false && !locks.has(id)) {
					// Register the starting run and invoke its durable append before this
					// call yields. A same-tick second send now sees the placeholder mailbox
					// and takes the durable queue path instead of living only in a lock.
					const mailbox = createMailbox();
					const controller = new AbortController();
					let resolveLive!: (result: RunResult) => void;
					let rejectLive!: (error: unknown) => void;
					const liveResult = new Promise<RunResult>((resolve, reject) => {
						resolveLive = resolve;
						rejectLive = reject;
					});
					void liveResult.catch(() => {});
					liveRuns.set(id, { mailbox, result: liveResult, controller });
					const messageId = newId();
					const acceptedAt = new Date().toISOString();
					const initialAppend = Promise.resolve(
						storage.append(id, {
							type: "user-message",
							at: acceptedAt,
							message,
							id: messageId,
							meta: sendOptions.meta,
						}),
					);
					const result = withSessionLock(id, async () => {
						await initialAppend;
						// Registered only after the durable append: an unpersisted send is
						// never reported accepted. execRun flushes the hook once the run id
						// exists, before any stream part.
						if (sendOptions.onAccepted) {
							const notify = sendOptions.onAccepted;
							mailbox.acceptHooks.push((runId) => {
								try {
									notify({ messageId, runId, queued: false, at: acceptedAt });
								} catch {
									// contained, like onPart: a broken listener must never break a send
								}
							});
						}
						const events = await storage.load(id);
						const inputIndex = events.findIndex(
							(event) => event.type === "user-message" && event.id === messageId,
						);
						const priorEvents = inputIndex >= 0 ? events.slice(0, inputIndex) : events;
						const prior = replaySession(priorEvents);
						const finalized = await finalizePersistedRun(id, agent, priorEvents, prior);
						if (prior.interruptedRunId && !finalized) {
							await storage.append(id, {
								type: "run-resume",
								at: new Date().toISOString(),
								runId: prior.interruptedRunId,
							});
						}
						return execRun<TOOLS>(
							id,
							agent,
							{
								abortSignal: sendOptions.abortSignal,
								onPart: sendOptions.onPart,
								onProgress: sendOptions.onProgress,
								resumeRunId:
									prior.interruptedRunId && !finalized ? prior.interruptedRunId : undefined,
								messageId,
							},
							mailbox,
							controller,
						);
					});
					result.then(resolveLive, rejectLive);
					const cleanup = () => {
						if (liveRuns.get(id)?.mailbox === mailbox) liveRuns.delete(id);
					};
					result.then(cleanup, cleanup);
					return result;
				}
				// queue:false always serializes. A default send also takes this path
				// when another API operation already reserved the session lock: its
				// input must not be appended early and then absorbed by that operation
				// before this send starts a duplicate run.
				return withSessionLock(id, async () => {
					const priorEvents = await storage.load(id);
					const prior = replaySession(priorEvents);
					const finalized = await finalizePersistedRun(id, agent, priorEvents, prior);
					const messageId = newId();
					const acceptedAt = new Date().toISOString();
					await storage.append(id, {
						type: "user-message",
						at: acceptedAt,
						message,
						id: messageId,
						meta: sendOptions.meta,
					});
					if (prior.interruptedRunId && !finalized) {
						await storage.append(id, {
							type: "run-resume",
							at: new Date().toISOString(),
							runId: prior.interruptedRunId,
						});
					}
					const mailbox = createMailbox();
					if (sendOptions.onAccepted) {
						const notify = sendOptions.onAccepted;
						mailbox.acceptHooks.push((runId) => {
							try {
								notify({ messageId, runId, queued: false, at: acceptedAt });
							} catch {
								// contained, like onPart: a broken listener must never break a send
							}
						});
					}
					return execRun<TOOLS>(
						id,
						agent,
						{
							abortSignal: sendOptions.abortSignal,
							onPart: sendOptions.onPart,
							onProgress: sendOptions.onProgress,
							resumeRunId:
								prior.interruptedRunId && !finalized ? prior.interruptedRunId : undefined,
							messageId,
						},
						mailbox,
					);
				});
			},
			attach(listener) {
				let listeners = attachedListeners.get(id);
				if (!listeners) {
					listeners = new Set();
					attachedListeners.set(id, listeners);
				}
				// A per-attach delegate, so attaching the same function twice yields
				// two independent subscriptions with independent detach handles.
				const entry = (event: ProgressEvent) => listener(event);
				listeners.add(entry);
				return () => {
					const current = attachedListeners.get(id);
					if (!current) return;
					current.delete(entry);
					if (current.size === 0) attachedListeners.delete(id);
				};
			},
			resume(resumeOptions = {}) {
				return withSessionLock(id, async () => {
					let events = await storage.load(id);
					let replayed = replaySession(events);
					if (!replayed.interruptedRunId && replayed.pendingMessages === 0) return null;
					const finalized = await finalizePersistedRun(id, agent, events, replayed);
					if (finalized) {
						if (replayed.pendingMessages === 0) return finalized;
						events = await storage.load(id);
						replayed = replaySession(events);
					}
					await storage.append(id, {
						type: "run-resume",
						at: new Date().toISOString(),
						runId: replayed.interruptedRunId,
					});
					return execRun<TOOLS>(id, agent, {
						abortSignal: resumeOptions.abortSignal,
						onPart: resumeOptions.onPart,
						onProgress: resumeOptions.onProgress,
						resumeRunId: replayed.interruptedRunId ?? undefined,
						messageId: latestInputMessageId(events, replayed),
					});
				});
			},
			async messages() {
				return replaySession(await storage.load(id)).messages;
			},
			transcript(transcriptOptions = {}) {
				return transcriptOf(id, transcriptOptions);
			},
			async stats() {
				const replayed = replaySession(await storage.load(id));
				return {
					...replayed.totals,
					contextWindow: await getContextWindow(agent.model),
				};
			},
			async isInterrupted() {
				if (liveRuns.has(id)) return false; // live in this process = running, not interrupted
				const replayed = replaySession(await storage.load(id));
				return replayed.interruptedRunId !== null || replayed.pendingMessages > 0;
			},
		};
	}

	async function task<T = string, TOOLS extends ToolSet = ToolSet>(
		taskOptions: TaskOptions<T, TOOLS>,
	): Promise<TaskOutcome<T>> {
		const sessionId = taskOptions.id ?? `task-${newId()}`;
		const taskStorage = taskOptions.durable === false ? memoryStorage() : storage;
		const schema = taskOptions.deliverable;

		let settled: TaskOutcome<T> | null = null;
		const settle = (outcome: TaskOutcome<T>) => {
			settled = outcome;
		};

		const emptyTotalsRef: UsageTotals = {
			inputTokens: 0,
			outputTokens: 0,
			totalTokens: 0,
			cachedInputTokens: 0,
			reasoningTokens: 0,
			cost: 0,
			steps: 0,
			contextTokens: null,
			contextWindow: null,
		};

		const schemaDescription = schema
			? (() => {
					try {
						return `\nThe deliverable MUST match this JSON schema:\n${JSON.stringify(z.toJSONSchema(schema as ZodType<unknown>))}`;
					} catch {
						return "";
					}
				})()
			: "";

		const taskTools = {
			submit_deliverable: tool({
				description:
					`Submit the final deliverable for this task. Calling this tool is the ONLY way to complete the task.` +
					schemaDescription +
					(taskOptions.deliverableHint ? `\n${taskOptions.deliverableHint}` : ""),
				inputSchema: z.object({
					deliverable: z
						.unknown()
						.describe(
							schema
								? "The deliverable, matching the required schema (pass the JSON value itself, not a string of it)."
								: "The deliverable.",
						),
				}),
				execute: async ({ deliverable }: { deliverable: unknown }) => {
					let value: unknown = deliverable;
					// Models often stringify JSON deliverables — unwrap before validating.
					if (schema && typeof value === "string") {
						try {
							value = JSON.parse(value);
						} catch {
							// leave as-is; the schema may legitimately want a string
						}
					}
					if (schema) {
						const check = schema.safeParse(value);
						if (!check.success) {
							return {
								accepted: false,
								error: `Deliverable rejected — schema validation failed: ${check.error.issues
									.map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`)
									.join("; ")}. Fix the deliverable and call submit_deliverable again.`,
							};
						}
						value = check.data;
					}
					settle({
						status: "submitted",
						deliverable: value as T,
						totals: emptyTotalsRef,
						sessionId,
					});
					return { accepted: true };
				},
			}),
			cancel_task: tool({
				description:
					"Cancel this task. Use when the task is impossible, malformed, unsafe, or asks for something you cannot or should not do. Explain why.",
				inputSchema: z.object({ reason: z.string() }),
				execute: async ({ reason }: { reason: string }) => {
					settle({ status: "cancelled", reason, totals: emptyTotalsRef, sessionId });
					return { ok: true };
				},
			}),
		};

		const systemSuffix =
			"\n\nThis is a TASK with a required outcome. You MUST end by calling submit_deliverable (when done) or cancel_task (when impossible or not allowed). Never end your turn without calling one of them. If submit_deliverable rejects your deliverable, fix it and submit again.";

		const agent: AgentConfig<ToolSet> = {
			...(taskOptions.agent as unknown as AgentConfig<ToolSet>),
			system: (taskOptions.agent.system ?? "You are a diligent task worker.") + systemSuffix,
			tools: { ...(taskOptions.agent.tools ?? {}), ...taskTools },
		};

		return withSessionLock(sessionId, async () => {
			// A resumed task may already be settled in its persisted history
			// (process died between the terminal tool running and run-end).
			const priorEvents = await taskStorage.load(sessionId);
			const prior = replaySession(priorEvents);
			const priorOutcome = findSettledOutcome<T>(
				prior.messages.map((entry) => entry.message),
				schema,
				sessionId,
			);
			if (priorOutcome) {
				if (prior.interruptedRunId) {
					const status = priorOutcome.status === "cancelled" ? "cancelled" : "completed";
					const error =
						priorOutcome.status === "cancelled" ? serializeError(priorOutcome.reason) : undefined;
					await taskStorage.append(sessionId, {
						type: "run-end",
						at: new Date().toISOString(),
						runId: prior.interruptedRunId,
						status,
						...(error ? { error } : {}),
					});
					emitEvent({
						type: "run-end",
						at: new Date().toISOString(),
						sessionId,
						runId: prior.interruptedRunId,
						status,
						totals: prior.totals,
						...(error ? { error } : {}),
					});
				}
				return { ...priorOutcome, totals: prior.totals };
			}
			const savedCancellation = latestExplicitCancellation(priorEvents);
			if (savedCancellation && savedCancellation.runId !== prior.interruptedRunId) {
				return {
					status: "cancelled",
					reason: savedCancellation.error.message,
					totals: prior.totals,
					sessionId,
				};
			}
			if (prior.interruptedRunId && interruptedStepForRun(priorEvents, prior.interruptedRunId)) {
				const finalized = await finalizePersistedRun(
					sessionId,
					agent,
					priorEvents,
					prior,
					taskStorage,
				);
				if (finalized?.status === "cancelled") {
					return {
						status: "cancelled",
						reason: finalized.error?.message ?? "Cancelled",
						totals: finalized.totals,
						sessionId,
					};
				}
			}

			let promptMessageId: string | undefined;
			if (prior.messages.length === 0) {
				const prompts =
					typeof taskOptions.prompt === "string"
						? [toUserMessage(taskOptions.prompt)]
						: taskOptions.prompt;
				for (const message of prompts) {
					promptMessageId = newId();
					await taskStorage.append(sessionId, {
						type: "user-message",
						at: new Date().toISOString(),
						message,
						id: promptMessageId,
					});
				}
			}
			if (prior.interruptedRunId) {
				await taskStorage.append(sessionId, {
					type: "run-resume",
					at: new Date().toISOString(),
					runId: prior.interruptedRunId,
				});
			}

			const result = await execRun<ToolSet>(
				sessionId,
				agent,
				{
					abortSignal: taskOptions.abortSignal,
					onPart: taskOptions.onPart as (
						part: TextStreamPart<ToolSet>,
						context: StreamContext,
					) => void,
					onProgress: taskOptions.onProgress,
					isSettled: () => settled !== null,
					maxPokes: taskOptions.maxPokes ?? 2,
					resumeRunId: prior.interruptedRunId ?? undefined,
					messageId: promptMessageId ?? latestInputMessageId(priorEvents, prior),
				},
				undefined,
				undefined,
				taskStorage,
			);

			if (settled) {
				const outcome = settled as TaskOutcome<T>;
				return { ...outcome, totals: result.totals };
			}
			if (result.status === "cancelled") {
				return {
					status: "cancelled",
					reason: result.error?.message ?? "Cancelled",
					totals: result.totals,
					sessionId,
				};
			}
			return {
				status: "failed",
				error: result.error ?? serializeError("Task ended without an outcome"),
				totals: result.totals,
				sessionId,
			};
		});
	}

	async function interruptedSessions(): Promise<string[]> {
		const ids = (await storage.listSessions?.()) ?? [];
		const interrupted: string[] = [];
		for (const id of ids) {
			// A run that is live in THIS process isn't interrupted — it's running.
			if (liveRuns.has(id)) continue;
			const replayed = replaySession(await storage.load(id));
			if (replayed.interruptedRunId || replayed.pendingMessages > 0) interrupted.push(id);
		}
		return interrupted;
	}

	const autoResume: AutoResumeOptions | undefined =
		typeof options.autoResume === "function"
			? { agentFor: options.autoResume }
			: options.autoResume;
	// Shared by boot and on-demand sweeps so racing resumeInterrupted() calls
	// cannot multiply the configured provider concurrency.
	const activeAutoResumeKicks = new Set<Promise<void>>();

	// The boot sweep (halo-style): find interrupted work, re-supply configs
	// through the resolver, and re-drive each session. Kicks are concurrency-
	// capped and optionally staggered; the returned promise settles when every
	// resumed run finishes. The per-session attempt cap lives in the LEDGER, so
	// a run that crashes the process on every resume stops being retried after
	// maxAttempts restarts instead of wedging the server into a boot loop.
	async function resumeInterrupted(): Promise<RunResult[]> {
		if (!autoResume) {
			throw new Error("resumeInterrupted() needs createAgentic({ autoResume: ... }) configured");
		}
		const maxAttempts = autoResume.maxAttempts ?? 3;
		const maxConcurrent = autoResume.maxConcurrent ?? 4;
		if (!Number.isInteger(maxConcurrent) || maxConcurrent < 1) {
			throw new RangeError("autoResume.maxConcurrent must be a positive integer");
		}
		const staggerMs = autoResume.staggerMs ?? 0;
		const kicks: Promise<RunResult | null>[] = [];
		for (const id of await interruptedSessions()) {
			const replayed = replaySession(await storage.load(id));
			const attempt = replayed.autoResumeAttempts + 1;
			if (attempt > maxAttempts) {
				await withSessionLock(id, async () => {
					if (liveRuns.has(id)) return;
					const current = replaySession(await storage.load(id));
					const currentAttempt = current.autoResumeAttempts + 1;
					if (currentAttempt <= maxAttempts) return;
					emitEvent({
						type: "auto-resume",
						at: new Date().toISOString(),
						sessionId: id,
						runId: current.interruptedRunId,
						attempt: currentAttempt,
						maxAttempts,
						action: "give-up",
					});
					if (!current.interruptedRunId) return;
					const error = serializeError(`Automatic resume attempt limit exhausted (${maxAttempts})`);
					await storage.append(id, {
						type: "run-end",
						at: new Date().toISOString(),
						runId: current.interruptedRunId,
						status: "failed",
						error,
					});
					emitEvent({
						type: "run-end",
						at: new Date().toISOString(),
						sessionId: id,
						runId: current.interruptedRunId,
						status: "failed",
						totals: current.totals,
						error,
					});
				});
				continue;
			}
			let agent: AgentConfig<ToolSet> | null | undefined;
			try {
				agent = await autoResume.agentFor(id);
			} catch {
				agent = null; // a broken resolver skips the session, never kills the sweep
			}
			if (!agent) continue;
			while (activeAutoResumeKicks.size >= maxConcurrent) {
				await Promise.race(activeAutoResumeKicks);
			}
			emitEvent({
				type: "auto-resume",
				at: new Date().toISOString(),
				sessionId: id,
				runId: replayed.interruptedRunId,
				attempt,
				maxAttempts,
				action: "resume",
			});
			const resolvedAgent = agent;
			const kick = withSessionLock(id, async () => {
				// Re-check under the lock — a send() may have raced the sweep.
				if (liveRuns.has(id)) return null;
				let events = await storage.load(id);
				let current = replaySession(events);
				if (!current.interruptedRunId && current.pendingMessages === 0) return null;
				const finalized = await finalizePersistedRun(id, resolvedAgent, events, current);
				if (finalized) {
					if (current.pendingMessages === 0) return finalized;
					events = await storage.load(id);
					current = replaySession(events);
				}
				await storage.append(id, {
					type: "run-resume",
					at: new Date().toISOString(),
					runId: current.interruptedRunId,
					auto: true,
				});
				return execRun<ToolSet>(id, resolvedAgent, {
					resumeRunId: current.interruptedRunId ?? undefined,
					messageId: latestInputMessageId(events, current),
				});
			});
			kicks.push(kick);
			const tracked = kick.then(
				() => undefined,
				() => undefined,
			);
			activeAutoResumeKicks.add(tracked);
			void tracked.then(() => activeAutoResumeKicks.delete(tracked));
			if (staggerMs > 0) await wait(staggerMs);
		}
		return (await Promise.all(kicks)).filter((r): r is RunResult => r !== null);
	}

	// Crash recovery that "just works": sweep in the background the moment
	// the harness comes up. Deferred a microtask so the caller finishes wiring
	// (event listeners etc.) before any run starts.
	if (autoResume) {
		queueMicrotask(() => {
			resumeInterrupted().catch(() => {
				// storage failures at boot must not take the process down;
				// the next resumeInterrupted() call (or restart) retries
			});
		});
	}

	return {
		session,
		task,
		async sessions() {
			return (await storage.listSessions?.()) ?? [];
		},
		transcript: transcriptOf,
		interruptedSessions,
		resumeInterrupted,
		isRunning(sessionId) {
			return liveRuns.has(sessionId);
		},
		cancel(sessionId, reason) {
			const live = liveRuns.get(sessionId);
			if (!live || live.controller.signal.aborted) return false;
			live.controller.abort(reason ?? new Error("Cancelled"));
			return true;
		},
	};
}

// RunResult.messageId for non-send entry points (resume, tasks, reconciled
// runs): the newest unanswered input when there is one, else the interrupted
// run's recorded initiating input, else the newest user message in the ledger.
function latestInputMessageId(events: StoredEvent[], replayed: ReplayedSession): string {
	const pending = replayed.pendingInputMessages.at(-1);
	if (pending) return pending.id;
	if (replayed.interruptedRunId) {
		for (const event of events) {
			if (
				event.type === "step" &&
				event.runId === replayed.interruptedRunId &&
				event.inputMessageIds.length > 0
			) {
				return event.inputMessageIds.at(-1) as string;
			}
		}
	}
	for (let index = events.length - 1; index >= 0; index -= 1) {
		const event = events[index];
		if (event.type === "user-message") return event.id;
	}
	return "";
}

// Which run persisted the first step that causally included this queued
// message. Sequence is insufficient: an in-flight step can land after the
// user-message event without having seen it.
function queueAnsweredByRun(events: StoredEvent[], queueId: string): string | null {
	for (const event of events) {
		if (event.type === "step" && event.acknowledgesInput && event.inputQueueIds.includes(queueId)) {
			return event.runId;
		}
	}
	return null;
}

function terminalEventForRun(
	events: StoredEvent[],
	runId: string,
): Extract<StoredEvent, { type: "run-end" }> | null {
	let terminal: Extract<StoredEvent, { type: "run-end" }> | null = null;
	for (const event of events) {
		if (event.type === "run-end" && event.runId === runId) terminal = event;
	}
	return terminal;
}

function lastAssistantTextForRun(events: StoredEvent[], runId: string): string {
	return lastAssistantText(
		events.flatMap((event) =>
			event.type === "step" && event.runId === runId
				? event.messages.map((stored) => stored.message)
				: [],
		),
	);
}

function finalizableStep(
	events: StoredEvent[],
	runId: string,
): Extract<StoredEvent, { type: "step" }> | null {
	let last: Extract<StoredEvent, { type: "step" }> | null = null;
	for (const event of events) {
		if (event.type === "step" && event.runId === runId) last = event;
	}
	return last &&
		!last.interrupted &&
		last.finishReason !== "tool-calls" &&
		last.finishReason !== "error" &&
		last.finishReason !== "cancelled"
		? last
		: null;
}

function interruptedStepForRun(
	events: StoredEvent[],
	runId: string,
): Extract<StoredEvent, { type: "step" }> | null {
	let last: Extract<StoredEvent, { type: "step" }> | null = null;
	for (const event of events) {
		if (event.type === "step" && event.runId === runId) last = event;
	}
	return last?.interrupted ? last : null;
}

function latestExplicitCancellation(events: StoredEvent[]): {
	runId: string;
	error: NonNullable<Extract<StoredEvent, { type: "step" }>["interrupted"]>["error"];
} | null {
	let runId: string | null = null;
	for (const event of events) {
		if (event.type === "run-start") runId = event.runId;
	}
	if (!runId) return null;
	const interrupted = interruptedStepForRun(events, runId)?.interrupted;
	if (!interrupted) return null;
	const terminal = terminalEventForRun(events, runId);
	if (terminal && terminal.status !== "cancelled") return null;
	return { runId, error: interrupted.error };
}

function lastAssistantText(messages: ModelMessage[]): string {
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];
		if (message.role !== "assistant") continue;
		const text = textOf(message);
		if (text.trim()) return text;
	}
	return "";
}

// A task outcome recovered from persisted history. Totals are deliberately
// absent — a history scan cannot know them; the caller attaches the replayed
// session's totals when it builds the full TaskOutcome.
type SettledTaskOutcome<T> =
	| { status: "submitted"; deliverable: T; sessionId: string }
	| { status: "cancelled"; reason: string; sessionId: string };

// Scan a replayed conversation for an already-settled task outcome: a
// submit_deliverable tool result with { accepted: true } (deliverable comes
// from the paired tool-call input) or a cancel_task result.
function findSettledOutcome<T>(
	messages: ModelMessage[],
	schema: ZodType<T> | undefined,
	sessionId: string,
): SettledTaskOutcome<T> | null {
	const callInputs = new Map<string, unknown>();
	for (const message of messages) {
		if (!Array.isArray(message.content)) continue;
		for (const part of message.content as Array<Record<string, unknown>>) {
			if (part.type === "tool-call" && typeof part.toolCallId === "string") {
				callInputs.set(part.toolCallId, part.input);
			}
			if (part.type !== "tool-result" || typeof part.toolCallId !== "string") continue;
			const output = part.output as { type?: string; value?: unknown } | undefined;
			const value = output?.type === "json" ? (output.value as Record<string, unknown>) : undefined;
			if (part.toolName === "submit_deliverable" && value?.accepted === true) {
				const input = callInputs.get(part.toolCallId) as { deliverable?: unknown } | undefined;
				let deliverable: unknown = input?.deliverable;
				if (schema && typeof deliverable === "string") {
					try {
						deliverable = JSON.parse(deliverable);
					} catch {
						// string deliverable
					}
				}
				if (schema) {
					const check = schema.safeParse(deliverable);
					if (!check.success) continue;
					deliverable = check.data;
				}
				return { status: "submitted", deliverable: deliverable as T, sessionId };
			}
			if (part.toolName === "cancel_task" && value?.ok === true) {
				const input = callInputs.get(part.toolCallId) as { reason?: string } | undefined;
				return { status: "cancelled", reason: input?.reason ?? "Cancelled", sessionId };
			}
		}
	}
	return null;
}

// ── one-shot resilience for plain generateText/streamText calls ─────────

export interface WithRetriesOptions {
	maxAttempts?: number;
	baseDelayMs?: number;
	maxDelayMs?: number;
	abortSignal?: AbortSignal;
	onRetry?: (info: { attempt: number; delayMs: number; error: unknown }) => void;
}

/**
 * Retry an arbitrary model call on transient failures (same classifier as
 * the agent loop). Fails fast on deterministic errors — billing, auth,
 * policy, context overflow.
 */
export async function withRetries<T>(
	fn: () => Promise<T>,
	options: WithRetriesOptions = {},
): Promise<T> {
	const retry = resolveRetryConfig({
		maxAttempts: options.maxAttempts,
		baseDelayMs: options.baseDelayMs,
		maxDelayMs: options.maxDelayMs,
	});
	let attempt = 0;
	while (true) {
		try {
			return await fn();
		} catch (err) {
			const cls = classifyFailure(err);
			if (
				cls.kind !== "transient" ||
				attempt >= retry.maxAttempts ||
				options.abortSignal?.aborted
			) {
				throw err;
			}
			attempt += 1;
			const delayMs = retryDelayMs(attempt, retry, cls.retryAfterMs);
			options.onRetry?.({ attempt, delayMs, error: err });
			await wait(delayMs, options.abortSignal);
		}
	}
}
