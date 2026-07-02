import {
	type LanguageModel,
	type ModelMessage,
	stepCountIs,
	streamText,
	type TextStreamPart,
	type ToolSet,
} from "ai";
import { resolveRetryConfig, retryDelayMs, wait } from "./backoff.js";
import { resolveCompactionConfig, runCompaction, shouldCompact } from "./compaction.js";
import { classifyFailure, serializeError } from "./failure.js";
import { getContextWindow } from "./modelMeta.js";
import { replaySession } from "./replay.js";
import { guardToolResultSizes } from "./toolGuard.js";
import type {
	AgentConfig,
	AgenticEvent,
	EventListener,
	RunResult,
	StepUsage,
	StorageProvider,
	StoredEvent,
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

export interface RunLoopOptions<TOOLS extends ToolSet = ToolSet> {
	sessionId: string;
	agent: AgentConfig<TOOLS>;
	storage: StorageProvider;
	/** Model factory (modelId → LanguageModel), wired to OpenRouter upstream. */
	// biome-ignore lint/suspicious/noExplicitAny: contravariant position — any AgentConfig works
	getModel: (modelId: string, agent: AgentConfig<any>) => LanguageModel;
	emit?: EventListener;
	abortSignal?: AbortSignal;
	/** Live stream parts (text deltas, tool calls…) for UIs. */
	onPart?: (part: TextStreamPart<TOOLS>) => void;
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
}

function newId(): string {
	return globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2);
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

	const emit = (event: AgenticEvent) => {
		try {
			options.emit?.(event);
		} catch {
			// a broken listener must never kill a run
		}
	};
	const append = (event: StoredEvent) => storage.append(sessionId, event);

	const runId = options.resumeRunId ?? newId();
	let runStarted = options.resumeRunId !== undefined;
	let totals: UsageTotals = emptyTotals();
	let finalText = "";
	let attempt = 0;
	let pokes = 0;
	let forceCompact = false;
	// Progress since the last forced compaction — if a context-overflow error
	// recurs with no new steps in between, compaction cannot save this run.
	let progressSinceForcedCompaction = true;
	const startedAt = Date.now();

	const fail = async (error: unknown): Promise<RunResult> => {
		const serialized = serializeError(error);
		await append({ type: "run-end", at: now(), runId, status: "failed", error: serialized });
		emit({ type: "run-end", sessionId, runId, status: "failed", totals, error: serialized });
		return { status: "failed", text: finalText, totals, error: serialized };
	};

	const cancel = async (reason: unknown): Promise<RunResult> => {
		const serialized = serializeError(reason ?? "Cancelled");
		await append({ type: "run-end", at: now(), runId, status: "cancelled", error: serialized });
		emit({ type: "run-end", sessionId, runId, status: "cancelled", totals, error: serialized });
		return { status: "cancelled", text: finalText, totals, error: serialized };
	};

	// Summarize-and-rebase the session. "skip" = nothing left to compact away.
	const compactNow = async (): Promise<
		{ outcome: "done" | "skip" } | { outcome: "failed"; error: unknown }
	> => {
		const replayed = replaySession(await storage.load(sessionId));
		if (replayed.messages.length <= compaction.keepRecent + 1) return { outcome: "skip" };
		try {
			const compacted = await runCompaction({
				messages: replayed.messages,
				model: options.getModel(compaction.model ?? agent.model, agent),
				config: compaction,
				abortSignal: options.abortSignal,
			});
			await append({
				type: "compaction",
				at: now(),
				messages: agent.preserveProviderOptions
					? compacted.messages
					: stripProviderOptions(compacted.messages),
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
		if (options.abortSignal?.aborted) return cancel(options.abortSignal.reason);

		const replayed = replaySession(await storage.load(sessionId));
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

		// ── one streamText pass ──
		let compactPending = false;
		let progressThisPass = false;
		let lastFinishReason = "";
		let streamError: unknown;
		// onStepFinish persistence is async; every pending append must settle
		// before the next pass replays the session, or the replay races the
		// writes it depends on.
		const pendingAppends: Promise<unknown>[] = [];

		const result = streamText({
			model: options.getModel(agent.model, agent),
			system: agent.system,
			messages: replayed.messages,
			tools,
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
				...(agent.stopWhen ?? []),
			],
			onStepFinish: (step) => {
				const usage: StepUsage = extractStepUsage({
					usage: step.usage,
					providerMetadata: step.providerMetadata as Record<string, unknown> | undefined,
				});
				const newMessages = step.response.messages.slice();
				pendingAppends.push(
					Promise.resolve(
						append({
							type: "step",
							at: now(),
							runId,
							messages: agent.preserveProviderOptions
								? newMessages
								: stripProviderOptions(newMessages),
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
				if (part.type === "error") streamError = part.error;
				try {
					options.onPart?.(part);
				} catch {
					// a broken part listener must never kill a run
				}
			}
		} catch (err) {
			if (streamError === undefined) streamError = err;
		}
		await Promise.allSettled(pendingAppends);

		if (options.abortSignal?.aborted) return cancel(options.abortSignal.reason);

		// ── pass ended with a provider/stream error ──
		if (streamError !== undefined) {
			const cls = classifyFailure(streamError);

			if (cls.kind === "context-overflow") {
				if (!progressSinceForcedCompaction) {
					return fail(streamError); // compaction already tried and didn't help
				}
				forceCompact = true;
				continue;
			}

			if (cls.kind === "fatal") return fail(streamError);

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
			await append({
				type: "user-message",
				at: now(),
				message: { role: "user", content },
				meta: { poke: pokes },
			});
			emit({ type: "poke", sessionId, runId, poke: pokes, maxPokes });
			continue;
		}

		await append({ type: "run-end", at: now(), runId, status: "completed" });
		emit({ type: "run-end", sessionId, runId, status: "completed", totals });
		// The model finished its turn over the threshold — compact silently
		// AFTER the run so the next turn starts fresh, instead of re-entering
		// the model just to regenerate an answer it already gave. A failure
		// here is non-fatal: the threshold is still crossed, so the next run
		// retries it.
		if (compactPending) await compactNow();
		return { status: "completed", text: finalText, totals };
	}
}
