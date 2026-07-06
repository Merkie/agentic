import type { LanguageModel, ModelMessage, TextStreamPart, ToolSet } from "ai";
import { tool } from "ai";
import { type ZodType, z } from "zod";
import { resolveRetryConfig, retryDelayMs, wait } from "./backoff.js";
import { classifyFailure, serializeError } from "./failure.js";
import { logEvents } from "./logEvents.js";
import { getContextWindow } from "./modelMeta.js";
import { createOpenRouter } from "./openrouter.js";
import { replaySession } from "./replay.js";
import { createResilientFetch, type ResilientFetchOptions } from "./resilientFetch.js";
import { createMailbox, type RunLoopOptions, runLoop } from "./run.js";
import { fileStorage } from "./storage.js";
import type {
	AgentConfig,
	EventListener,
	MaybePromise,
	RunResult,
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
	/** Pause between resume kicks so a big boot doesn't stampede the provider. Default 0. */
	staggerMs?: number;
}

export interface AgenticOptions {
	/** Defaults to OPENROUTER_API_KEY from the environment. */
	apiKey?: string;
	/** Where sessions live. Defaults to JSONL files under ./.agentic */
	storage?: StorageProvider;
	/**
	 * Make crash recovery automatic: as soon as the harness is created it
	 * sweeps storage for interrupted work (open runs, unanswered queued
	 * messages) and resumes each session in the background, using this
	 * resolver to map session ids back to agent configs. Pass the resolver
	 * directly or wrap it in {@link AutoResumeOptions} to tune the attempt
	 * cap / stagger. resumeInterrupted() runs the same sweep on demand.
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
	onPart?: (part: TextStreamPart<TOOLS>) => void;
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
	resume(options?: Pick<SendOptions<TOOLS>, "onPart" | "abortSignal">): Promise<RunResult | null>;
	/** The replay-ready conversation as the model would see it. */
	messages(): Promise<ModelMessage[]>;
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
	/** Extra instructions appended to the submit tool's description. */
	deliverableHint?: string;
	maxPokes?: number;
	abortSignal?: AbortSignal;
	onPart?: (part: TextStreamPart<TOOLS>) => void;
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
	/** The underlying storage, for app-level queries. */
	storage: StorageProvider;
}

function toUserMessage(content: string | ModelMessage): ModelMessage {
	return typeof content === "string" ? { role: "user", content } : content;
}

function newId(): string {
	return globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2);
}

export function createAgentic(options: AgenticOptions = {}): Agentic {
	const storage = options.storage ?? fileStorage();
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
		locks.set(
			sessionId,
			next.catch(() => {}),
		);
		return next;
	}

	// The session's live run, if any — what send() queues messages into.
	interface LiveRun {
		mailbox: ReturnType<typeof createMailbox>;
		result: Promise<RunResult>;
	}
	const liveRuns = new Map<string, LiveRun>();

	// Start a run registered in the live-run map so concurrent send()s can
	// queue into it. Must be called while holding the session lock.
	function execRun<TOOLS extends ToolSet>(
		sessionId: string,
		agent: AgentConfig<TOOLS>,
		runOptions: Partial<
			Pick<
				RunLoopOptions<TOOLS>,
				"abortSignal" | "onPart" | "resumeRunId" | "isSettled" | "pokeMessage" | "maxPokes"
			>
		>,
	): Promise<RunResult> {
		const mailbox = createMailbox();
		const result = runLoop<TOOLS>({
			sessionId,
			agent,
			storage,
			getModel,
			emit: emitEvent,
			mailbox,
			...runOptions,
		});
		liveRuns.set(sessionId, { mailbox, result });
		const cleanup = () => {
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

	function session<TOOLS extends ToolSet>(id: string, agent: AgentConfig<TOOLS>): Session<TOOLS> {
		return {
			id,
			send(content, sendOptions = {}) {
				const message = toUserMessage(content);
				const live = sendOptions.queue !== false ? liveRuns.get(id) : undefined;
				if (live?.mailbox.accepting) {
					return (async () => {
						// Persist FIRST — the message survives even if everything
						// after this line dies. Then signal the live run.
						const queueId = newId();
						await storage.append(id, {
							type: "user-message",
							at: new Date().toISOString(),
							message,
							meta: { ...sendOptions.meta, queued: true, queueId },
						});
						if (live.mailbox.tryEnqueue(queueId)) {
							if (sendOptions.onPart) {
								live.mailbox.partListeners.push(sendOptions.onPart as (part: unknown) => void);
							}
							emitEvent({ type: "queued-message", sessionId: id, runId: live.mailbox.runId });
							return live.result;
						}
						// The run committed to ending while the append settled. The
						// message is safely in the ledger — give it its own run,
						// unless the ending run's final pass already replayed it.
						return withSessionLock(id, async () => {
							const events = await storage.load(id);
							if (queuedMessageAnswered(events, queueId)) {
								const replayed = replaySession(events);
								return {
									status: "completed" as const,
									text: lastAssistantText(replayed.messages),
									totals: {
										...replayed.totals,
										contextWindow: await getContextWindow(agent.model),
									},
								};
							}
							return execRun<TOOLS>(id, agent, {
								abortSignal: sendOptions.abortSignal,
								onPart: sendOptions.onPart,
							});
						});
					})();
				}
				return withSessionLock(id, async () => {
					await storage.append(id, {
						type: "user-message",
						at: new Date().toISOString(),
						message,
						meta: sendOptions.meta,
					});
					return execRun<TOOLS>(id, agent, {
						abortSignal: sendOptions.abortSignal,
						onPart: sendOptions.onPart,
					});
				});
			},
			resume(resumeOptions = {}) {
				return withSessionLock(id, async () => {
					const replayed = replaySession(await storage.load(id));
					if (!replayed.interruptedRunId && replayed.pendingMessages === 0) return null;
					await storage.append(id, {
						type: "run-resume",
						at: new Date().toISOString(),
						runId: replayed.interruptedRunId,
					});
					return execRun<TOOLS>(id, agent, {
						abortSignal: resumeOptions.abortSignal,
						onPart: resumeOptions.onPart,
						resumeRunId: replayed.interruptedRunId ?? undefined,
					});
				});
			},
			async messages() {
				return replaySession(await storage.load(id)).messages;
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
			const prior = replaySession(await storage.load(sessionId));
			const priorOutcome = findSettledOutcome<T>(prior.messages, schema, sessionId);
			if (priorOutcome) return { ...priorOutcome, totals: prior.totals };

			if (prior.messages.length === 0) {
				const prompts =
					typeof taskOptions.prompt === "string"
						? [toUserMessage(taskOptions.prompt)]
						: taskOptions.prompt;
				for (const message of prompts) {
					await storage.append(sessionId, {
						type: "user-message",
						at: new Date().toISOString(),
						message,
					});
				}
			}

			const result = await execRun<ToolSet>(sessionId, agent, {
				abortSignal: taskOptions.abortSignal,
				onPart: taskOptions.onPart as (part: TextStreamPart<ToolSet>) => void,
				isSettled: () => settled !== null,
				maxPokes: taskOptions.maxPokes ?? 2,
				resumeRunId: prior.interruptedRunId ?? undefined,
			});

			if (settled) {
				const outcome = settled as TaskOutcome<T>;
				return { ...outcome, totals: result.totals };
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

	// The boot sweep (halo-style): find interrupted work, re-supply configs
	// through the resolver, and re-drive each session. Kicks are concurrent
	// (optionally staggered); the returned promise settles when every resumed
	// run finishes. The per-session attempt cap lives in the LEDGER, so a run
	// that crashes the process on every resume stops being retried after
	// maxAttempts restarts instead of wedging the server into a boot loop.
	async function resumeInterrupted(): Promise<RunResult[]> {
		if (!autoResume) {
			throw new Error("resumeInterrupted() needs createAgentic({ autoResume: ... }) configured");
		}
		const maxAttempts = autoResume.maxAttempts ?? 3;
		const staggerMs = autoResume.staggerMs ?? 0;
		const kicks: Promise<RunResult | null>[] = [];
		for (const id of await interruptedSessions()) {
			const replayed = replaySession(await storage.load(id));
			const attempt = replayed.autoResumeAttempts + 1;
			if (attempt > maxAttempts) {
				emitEvent({
					type: "auto-resume",
					sessionId: id,
					runId: replayed.interruptedRunId,
					attempt,
					maxAttempts,
					action: "give-up",
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
			emitEvent({
				type: "auto-resume",
				sessionId: id,
				runId: replayed.interruptedRunId,
				attempt,
				maxAttempts,
				action: "resume",
			});
			const resolvedAgent = agent;
			kicks.push(
				withSessionLock(id, async () => {
					// Re-check under the lock — a send() may have raced the sweep.
					if (liveRuns.has(id)) return null;
					const current = replaySession(await storage.load(id));
					if (!current.interruptedRunId && current.pendingMessages === 0) return null;
					await storage.append(id, {
						type: "run-resume",
						at: new Date().toISOString(),
						runId: current.interruptedRunId,
						auto: true,
					});
					return execRun<ToolSet>(id, resolvedAgent, {
						resumeRunId: current.interruptedRunId ?? undefined,
					});
				}),
			);
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
		interruptedSessions,
		resumeInterrupted,
		storage,
	};
}

// Did any step land after this queued message? Used by the send() fallback
// when its enqueue lost the race with the run's exit: a step after the
// message means the ending run's final pass replayed (and answered) it, so
// starting another run would answer it twice.
function queuedMessageAnswered(events: StoredEvent[], queueId: string): boolean {
	let seen = false;
	for (const event of events) {
		if (event.type === "user-message" && event.meta?.queueId === queueId) {
			seen = true;
			continue;
		}
		if (seen && event.type === "step") return true;
	}
	return false;
}

function lastAssistantText(messages: ModelMessage[]): string {
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];
		if (message.role !== "assistant") continue;
		if (typeof message.content === "string") return message.content;
		return (message.content as Array<{ type?: string; text?: string }>)
			.filter((part) => part.type === "text")
			.map((part) => part.text ?? "")
			.join("");
	}
	return "";
}

// Scan a replayed conversation for an already-settled task outcome: a
// submit_deliverable tool result with { accepted: true } (deliverable comes
// from the paired tool-call input) or a cancel_task result.
function findSettledOutcome<T>(
	messages: ModelMessage[],
	schema: ZodType<T> | undefined,
	sessionId: string,
): TaskOutcome<T> | null {
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
				return {
					status: "submitted",
					deliverable: deliverable as T,
					totals: undefined as unknown as UsageTotals,
					sessionId,
				};
			}
			if (part.toolName === "cancel_task" && value?.ok === true) {
				const input = callInputs.get(part.toolCallId) as { reason?: string } | undefined;
				return {
					status: "cancelled",
					reason: input?.reason ?? "Cancelled",
					totals: undefined as unknown as UsageTotals,
					sessionId,
				};
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
