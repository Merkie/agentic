import type { LanguageModel, ModelMessage, TextStreamPart, ToolSet } from "ai";
import { tool } from "ai";
import { z, type ZodType } from "zod";
import { resolveRetryConfig, retryDelayMs, wait } from "./backoff.js";
import { classifyFailure, serializeError } from "./failure.js";
import { getContextWindow } from "./modelMeta.js";
import { createOpenRouter } from "./openrouter.js";
import { replaySession } from "./replay.js";
import { createResilientFetch, type ResilientFetchOptions } from "./resilientFetch.js";
import { fileStorage } from "./storage.js";
import { runLoop } from "./run.js";
import type {
	AgentConfig,
	EventListener,
	RunResult,
	StorageProvider,
	TaskOutcome,
	UsageTotals,
} from "./types.js";

export interface AgenticOptions {
	/** Defaults to OPENROUTER_API_KEY from the environment. */
	apiKey?: string;
	/** Where sessions live. Defaults to JSONL files under ./.agentic */
	storage?: StorageProvider;
	/** Observability firehose: run/step/retry/compaction/poke events. */
	onEvent?: EventListener;
	/** App attribution etc., sent on every request. */
	headers?: Record<string, string>;
	/** Provider routing etc., merged into every request body. */
	extraBody?: Record<string, unknown>;
	/** Header/SSE-idle stall detection. Defaults 60s/120s; false disables. */
	fetchTimeouts?: ResilientFetchOptions | false;
}

export interface SendOptions<TOOLS extends ToolSet = ToolSet> {
	onPart?: (part: TextStreamPart<TOOLS>) => void;
	abortSignal?: AbortSignal;
	/** App-supplied tag stored on the user-message event (dedup keys etc.). */
	meta?: Record<string, unknown>;
}

export interface Session<TOOLS extends ToolSet = ToolSet> {
	id: string;
	/** Append a user message and run the agent to completion. */
	send(content: string | ModelMessage, options?: SendOptions<TOOLS>): Promise<RunResult>;
	/**
	 * Re-enter a run that a crash/restart left open (run-start with no
	 * run-end). Resolves null when there is nothing to resume.
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
	/** Sessions with an interrupted run — feed these to resume() on boot. */
	interruptedSessions(): Promise<string[]>;
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
		openrouter.chat(modelId, agent.extraBody ? { extraBody: agent.extraBody } : {});

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

	function session<TOOLS extends ToolSet>(id: string, agent: AgentConfig<TOOLS>): Session<TOOLS> {
		return {
			id,
			send(content, sendOptions = {}) {
				return withSessionLock(id, async () => {
					await storage.append(id, {
						type: "user-message",
						at: new Date().toISOString(),
						message: toUserMessage(content),
						meta: sendOptions.meta,
					});
					return runLoop<TOOLS>({
						sessionId: id,
						agent,
						storage,
						getModel,
						emit: options.onEvent,
						abortSignal: sendOptions.abortSignal,
						onPart: sendOptions.onPart,
					});
				});
			},
			resume(resumeOptions = {}) {
				return withSessionLock(id, async () => {
					const replayed = replaySession(await storage.load(id));
					if (!replayed.interruptedRunId) return null;
					return runLoop<TOOLS>({
						sessionId: id,
						agent,
						storage,
						getModel,
						emit: options.onEvent,
						abortSignal: resumeOptions.abortSignal,
						onPart: resumeOptions.onPart,
						resumeRunId: replayed.interruptedRunId,
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
				return replaySession(await storage.load(id)).interruptedRunId !== null;
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

			const result = await runLoop<ToolSet>({
				sessionId,
				agent,
				storage,
				getModel,
				emit: options.onEvent,
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

	return {
		session,
		task,
		async sessions() {
			return (await storage.listSessions?.()) ?? [];
		},
		async interruptedSessions() {
			const ids = (await storage.listSessions?.()) ?? [];
			const interrupted: string[] = [];
			for (const id of ids) {
				if (replaySession(await storage.load(id)).interruptedRunId) interrupted.push(id);
			}
			return interrupted;
		},
		storage,
	};
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
			if (cls.kind !== "transient" || attempt >= retry.maxAttempts || options.abortSignal?.aborted) {
				throw err;
			}
			attempt += 1;
			const delayMs = retryDelayMs(attempt, retry, cls.retryAfterMs);
			options.onRetry?.({ attempt, delayMs, error: err });
			await wait(delayMs, options.abortSignal);
		}
	}
}
