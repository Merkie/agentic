import type { LanguageModelV3StreamPart } from "@ai-sdk/provider";
import { type ModelMessage, tool } from "ai";
import { convertArrayToReadableStream, MockLanguageModelV3 } from "ai/test";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { createAgentic } from "./agentic.js";
import { retryDelayMs } from "./backoff.js";
import { classifyFailure } from "./failure.js";
import { setContextWindow } from "./modelMeta.js";
import { replaySession } from "./replay.js";
import { createMailbox, hoistSandwichedUsers, runLoop } from "./run.js";
import { sanitizeConversation } from "./sanitize.js";
import { memoryStorage } from "./storage.js";
import type { AgenticEvent, StepUsage, StoredEvent, StoredMessage } from "./types.js";
import { extractStepUsage, reconcileBilledCost } from "./usage.js";

let storedSeq = 0;
const stored = (...messages: ModelMessage[]): StoredMessage[] =>
	messages.map((message) => ({ id: `sm-${++storedSeq}`, message }));

const usage = (partial: Partial<StepUsage> = {}): StepUsage => ({
	inputTokens: 100,
	outputTokens: 20,
	totalTokens: 120,
	cachedInputTokens: null,
	reasoningTokens: null,
	cost: 0.001,
	upstreamCost: null,
	isByok: null,
	billedCost: 0.001,
	...partial,
});

describe("classifyFailure", () => {
	it("classifies provider drops and rate limits as transient", () => {
		expect(classifyFailure(new Error("502 Upstream idle timeout exceeded")).kind).toBe("transient");
		expect(classifyFailure(new Error("ECONNRESET: socket hang up")).kind).toBe("transient");
		expect(classifyFailure({ message: "Rate limit exceeded", code: 429 }).kind).toBe("transient");
		expect(
			classifyFailure(new Error("No output generated. Check the stream for errors.")).kind,
		).toBe("transient");
	});

	it("classifies context overflow as its own kind, not transient", () => {
		expect(
			classifyFailure(new Error("This model's maximum context length is 65536 tokens")).kind,
		).toBe("context-overflow");
		expect(classifyFailure(new Error("prompt is too long: 210000 tokens")).kind).toBe(
			"context-overflow",
		);
	});

	it("does not confuse rate limits with overflow", () => {
		expect(classifyFailure(new Error("too many requests, rate limit hit")).kind).toBe("transient");
	});

	it("classifies billing/auth/malformed as fatal", () => {
		expect(
			classifyFailure(new Error("This request requires more credits, or fewer max_tokens")).kind,
		).toBe("fatal");
		expect(classifyFailure(new Error("Invalid API key")).kind).toBe("fatal");
		expect(
			classifyFailure(
				new Error("[Xiaomi] Param Incorrect: messages[3].tool_calls[0] is missing a function name"),
			).kind,
		).toBe("fatal");
	});

	it("prefers OpenRouter's typed error code when present", () => {
		expect(
			classifyFailure({ message: "x", metadata: { error_type: "provider_overloaded" } }).kind,
		).toBe("transient");
		expect(
			classifyFailure({ message: "x", metadata: { error_type: "content_policy_violation" } }).kind,
		).toBe("fatal");
	});

	it("extracts Retry-After for transient failures", () => {
		const err = Object.assign(new Error("429 too many requests"), {
			responseHeaders: { "retry-after": "7" },
		});
		expect(classifyFailure(err).retryAfterMs).toBe(7000);
	});
});

describe("retryDelayMs", () => {
	const cfg = { maxAttempts: 6, baseDelayMs: 2000, maxDelayMs: 60000, maxElapsedMs: 900000 };
	it("backs off exponentially with a cap", () => {
		expect(retryDelayMs(1, cfg)).toBe(2000);
		expect(retryDelayMs(2, cfg)).toBe(4000);
		expect(retryDelayMs(6, cfg)).toBe(60000);
	});
	it("lets Retry-After win over backoff", () => {
		expect(retryDelayMs(1, cfg, 30000)).toBe(30000);
		expect(retryDelayMs(1, cfg, 300000)).toBe(60000); // still capped
	});
});

describe("reconcileBilledCost (BYOK vs credits)", () => {
	it("sums fee + provider charge on BYOK", () => {
		expect(reconcileBilledCost(0, 0.0005, true)).toBe(0.0005); // BYOK, no fee
		expect(reconcileBilledCost(0.000025, 0.0005, true)).toBe(0.000525); // BYOK + 5% fee
		expect(reconcileBilledCost(null, 0.0005, true)).toBe(0.0005);
	});
	it("takes cost alone on credits — upstream is a mirror, not a charge", () => {
		expect(reconcileBilledCost(0.000003752, 0.000003752, false)).toBe(0.000003752);
		expect(reconcileBilledCost(0.001, null, false)).toBe(0.001);
	});
	it("never sums when is_byok is unknown", () => {
		expect(reconcileBilledCost(0.001, null)).toBe(0.001); // pre-is_byok credits
		expect(reconcileBilledCost(0.001, 0.001)).toBe(0.001); // mirrored payload, no flag
		expect(reconcileBilledCost(null, 0.0005)).toBe(0.0005);
		expect(reconcileBilledCost(null, null)).toBeNull();
	});
});

describe("extractStepUsage (is_byok classification)", () => {
	const raw = (over: Record<string, unknown>) =>
		({
			usage: {
				inputTokens: 249,
				outputTokens: 10,
				totalTokens: 259,
				raw: over,
			},
		}) as Parameters<typeof extractStepUsage>[0];

	it("reads is_byok from usage.raw and sums the BYOK charge", () => {
		const u = extractStepUsage(
			raw({ cost: 0, is_byok: true, cost_details: { upstream_inference_cost: 0.0000113176 } }),
		);
		expect(u.isByok).toBe(true);
		expect(u.billedCost).toBeCloseTo(0.0000113176, 12);
	});

	it("does not double-count a mirrored credits payload", () => {
		const u = extractStepUsage(
			raw({
				cost: 0.0000025872,
				is_byok: false,
				cost_details: { upstream_inference_cost: 0.0000025872 },
			}),
		);
		expect(u.isByok).toBe(false);
		expect(u.upstreamCost).toBeCloseTo(0.0000025872, 12); // kept as data
		expect(u.billedCost).toBeCloseTo(0.0000025872, 12); // charged once
	});

	it("treats a missing is_byok as unknown and never sums", () => {
		const u = extractStepUsage(
			raw({ cost: 0.001, cost_details: { upstream_inference_cost: 0.001 } }),
		);
		expect(u.isByok).toBeNull();
		expect(u.billedCost).toBe(0.001);
	});
});

describe("replaySession", () => {
	it("rebuilds messages, detects interrupted runs, resets context on compaction", () => {
		const events: StoredEvent[] = [
			{ type: "user-message", at: "t", id: "u-1", message: { role: "user", content: "hi" } },
			{ type: "run-start", at: "t", runId: "r1", model: "m" },
			{
				type: "step",
				at: "t",
				runId: "r1",
				messages: stored({ role: "assistant", content: "hello" }),
				inputMessageIds: ["u-1"],
				inputQueueIds: [],
				acknowledgesInput: true,
				finishReason: "stop",
				usage: usage(),
			},
			{ type: "run-end", at: "t", runId: "r1", status: "completed" },
			{
				type: "compaction",
				at: "t",
				messages: stored({ role: "user", content: "<summary>" }),
				usage: usage({ inputTokens: 5000, cost: 0.002, billedCost: 0.002 }),
			},
			{ type: "user-message", at: "t", id: "u-2", message: { role: "user", content: "again" } },
			{ type: "run-start", at: "t", runId: "r2", model: "m" },
		];
		const replayed = replaySession(events);
		expect(replayed.messages.map((m) => m.message.content)).toEqual(["<summary>", "again"]);
		expect(replayed.interruptedRunId).toBe("r2");
		expect(replayed.contextTokens).toBeNull(); // compaction reset, no step since
		expect(replayed.totals.cost).toBeCloseTo(0.003);
		expect(replayed.totals.steps).toBe(2);
	});

	it("counts trailing unanswered user messages as pending", () => {
		const step: StoredEvent = {
			type: "step",
			at: "t",
			runId: "r1",
			messages: stored({ role: "assistant", content: "hello" }),
			inputMessageIds: ["u-hi"],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "stop",
			usage: usage(),
		};
		const user = (content: string): StoredEvent => ({
			type: "user-message",
			at: "t",
			id: `u-${content}`,
			message: { role: "user", content },
		});

		// a send() that crashed before its run even started
		expect(replaySession([user("hi")]).pendingMessages).toBe(1);
		// answered by a step
		expect(
			replaySession([user("hi"), { type: "run-start", at: "t", runId: "r1", model: "m" }, step])
				.pendingMessages,
		).toBe(0);
		// queued message that landed after the run ended
		expect(
			replaySession([
				user("hi"),
				{ type: "run-start", at: "t", runId: "r1", model: "m" },
				step,
				{ type: "run-end", at: "t", runId: "r1", status: "completed" },
				user("one more thing"),
			]).pendingMessages,
		).toBe(1);
		// a cancelled run settles its messages — don't auto-resume an abort
		expect(
			replaySession([
				user("hi"),
				{ type: "run-start", at: "t", runId: "r1", model: "m" },
				{ type: "run-end", at: "t", runId: "r1", status: "cancelled" },
			]).pendingMessages,
		).toBe(0);
	});

	it("counts auto resume attempts, ignoring manual ones, reset by run-end", () => {
		const open: StoredEvent[] = [
			{ type: "run-start", at: "t", runId: "r1", model: "m" },
			{ type: "run-resume", at: "t", runId: "r1", auto: true },
			// manual resume() — uncapped, must not count toward the breaker
			{ type: "run-resume", at: "t", runId: "r1" },
			{ type: "run-resume", at: "t", runId: "r1", auto: true },
		];
		expect(replaySession(open).autoResumeAttempts).toBe(2);
		expect(
			replaySession([...open, { type: "run-end", at: "t", runId: "r1", status: "completed" }])
				.autoResumeAttempts,
		).toBe(0);
	});

	it("keeps a queued message pending until a causal step includes its queue id", () => {
		const events: StoredEvent[] = [
			{ type: "user-message", at: "t0", id: "u-a", message: { role: "user", content: "A" } },
			{ type: "run-start", at: "t1", runId: "r1", model: "m" },
			{
				type: "user-message",
				at: "t2",
				id: "q-b",
				message: { role: "user", content: "B" },
				meta: { queued: true, queueId: "q-b" },
			},
			{
				type: "step",
				at: "t3",
				runId: "r1",
				messages: stored({ role: "assistant", content: "answer to A" }),
				inputMessageIds: ["u-a"],
				inputQueueIds: [],
				acknowledgesInput: true,
				finishReason: "stop",
				usage: usage(),
			},
			{ type: "run-end", at: "t4", runId: "r1", status: "failed" },
		];

		const pending = replaySession(events);
		expect(pending.pendingQueueIds).toEqual(["q-b"]);
		expect(pending.pendingMessages).toBe(1);

		const answered = replaySession([
			...events,
			{ type: "run-start", at: "t5", runId: "r2", model: "m" },
			{
				type: "step",
				at: "t6",
				runId: "r2",
				messages: stored({ role: "assistant", content: "answer to B" }),
				inputMessageIds: ["q-b"],
				inputQueueIds: ["q-b"],
				acknowledgesInput: true,
				finishReason: "stop",
				usage: usage(),
			},
			{ type: "run-end", at: "t7", runId: "r2", status: "completed" },
		]);
		expect(answered.pendingQueueIds).toEqual([]);
		expect(answered.pendingMessages).toBe(0);
	});

	it("retains auto-resume attempts while failed queued work remains pending", () => {
		const replayed = replaySession([
			{
				type: "user-message",
				at: "t0",
				id: "q-1",
				message: { role: "user", content: "queued" },
				meta: { queued: true, queueId: "q-1" },
			},
			{ type: "run-resume", at: "t1", runId: null, auto: true },
			{ type: "run-start", at: "t2", runId: "r1", model: "m" },
			{ type: "run-end", at: "t3", runId: "r1", status: "failed" },
		]);

		expect(replayed.pendingQueueIds).toEqual(["q-1"]);
		expect(replayed.autoResumeAttempts).toBe(1);
	});

	it("does not treat an app-owned meta.queueId as framework queue state", () => {
		const replayed = replaySession([
			{
				type: "user-message",
				at: "t0",
				id: "u-1",
				message: { role: "user", content: "ordinary input" },
				meta: { queueId: "domain-id" },
			},
			{ type: "run-start", at: "t1", runId: "r1", model: "m" },
			{ type: "run-end", at: "t2", runId: "r1", status: "failed" },
		]);

		expect(replayed.pendingQueueIds).toEqual([]);
		expect(replayed.pendingMessages).toBe(0);
		expect(replayed.interruptedRunId).toBeNull();
	});
});

describe("sanitizeConversation", () => {
	it("drops dangling tool calls from an interrupted run", () => {
		const { messages, removed } = sanitizeConversation([
			{ role: "user", content: "go" },
			{
				role: "assistant",
				content: [
					{ type: "text", text: "on it" },
					{ type: "tool-call", toolCallId: "a", toolName: "lookup", input: {} },
				],
			},
			// process died before the tool result was persisted
		]);
		expect(removed).toBe(1);
		expect(messages).toHaveLength(2);
		expect((messages[1].content as { type: string }[]).map((p) => p.type)).toEqual(["text"]);
	});
});

describe("memoryStorage", () => {
	it("appends and loads in order", async () => {
		const storage = memoryStorage();
		await storage.append("s", {
			type: "user-message",
			at: "t",
			id: "u-1",
			message: { role: "user", content: "1" },
		});
		await storage.append("s", { type: "run-start", at: "t", runId: "r", model: "m" });
		const events = await storage.load("s");
		expect(events.map((e) => e.type)).toEqual(["user-message", "run-start"]);
	});
});

describe("validated tasks", () => {
	const finish = (reason: "stop" | "tool-calls" = "tool-calls"): LanguageModelV3StreamPart => ({
		type: "finish",
		finishReason: { unified: reason, raw: reason },
		usage: {
			inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
			outputTokens: { total: 5, text: 5, reasoning: undefined },
		},
	});

	it("returns schema errors to the model and accepts a corrected submission", async () => {
		setContextWindow("mock/task", 100_000);
		const prompts: unknown[] = [];
		let call = 0;
		const model = new MockLanguageModelV3({
			doStream: async ({ prompt }) => {
				prompts.push(prompt);
				call += 1;
				const deliverable =
					call === 1 ? { count: "not-a-number" } : { count: 3, label: "corrected" };
				return {
					stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
						{ type: "stream-start", warnings: [] },
						{
							type: "tool-call",
							toolCallId: `submit-${call}`,
							toolName: "submit_deliverable",
							input: JSON.stringify({ deliverable }),
						},
						finish(),
					]),
				};
			},
		});
		const storage = memoryStorage();
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
		});

		const outcome = await agentic.task({
			id: "validated-task",
			agent: { model: "mock/task" },
			prompt: "Return a count and label.",
			deliverable: z.object({ count: z.number(), label: z.string() }),
		});

		expect(outcome).toMatchObject({
			status: "submitted",
			deliverable: { count: 3, label: "corrected" },
		});
		expect(call).toBe(2);
		expect(JSON.stringify(prompts[1])).toContain("schema validation failed");
		expect(replaySession(await storage.load("validated-task")).pendingMessages).toBe(0);
	});

	it("replays a previously accepted outcome without calling the model again", async () => {
		setContextWindow("mock/task-replay", 100_000);
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				return {
					stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
						{ type: "stream-start", warnings: [] },
						{
							type: "tool-call",
							toolCallId: "submit-once",
							toolName: "submit_deliverable",
							input: JSON.stringify({ deliverable: { value: "kept" } }),
						},
						finish(),
					]),
				};
			},
		});
		const storage = memoryStorage();
		const agentic = createAgentic({ apiKey: "test", storage, getModel: () => model });
		const options = {
			id: "replayed-task",
			agent: { model: "mock/task-replay" },
			prompt: "Return a value.",
			deliverable: z.object({ value: z.string() }),
		};

		await expect(agentic.task(options)).resolves.toMatchObject({ status: "submitted" });
		await expect(agentic.task(options)).resolves.toMatchObject({
			status: "submitted",
			deliverable: { value: "kept" },
		});
		expect(calls).toBe(1);
	});

	it("can run a one-shot task without creating a durable session", async () => {
		setContextWindow("mock/ephemeral-task", 100_000);
		const model = new MockLanguageModelV3({
			doStream: async () => ({
				stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
					{ type: "stream-start", warnings: [] },
					{
						type: "tool-call",
						toolCallId: "submit-ephemeral",
						toolName: "submit_deliverable",
						input: JSON.stringify({ deliverable: { value: "temporary" } }),
					},
					finish(),
				]),
			}),
		});
		const storage = memoryStorage();
		const agentic = createAgentic({ storage, getModel: () => model });

		await expect(
			agentic.task({
				id: "ephemeral-task",
				durable: false,
				agent: { model: "mock/ephemeral-task" },
				prompt: "Return a value.",
				deliverable: z.object({ value: z.string() }),
			}),
		).resolves.toMatchObject({
			status: "submitted",
			deliverable: { value: "temporary" },
		});
		expect(await storage.listSessions?.()).toEqual([]);
	});

	it("resumes an interrupted task under the original run id", async () => {
		setContextWindow("mock/task-resume", 100_000);
		const storage = memoryStorage();
		await storage.append("interrupted-task", {
			type: "user-message",
			at: "t0",
			id: "u-1",
			message: { role: "user", content: "finish this" },
		});
		await storage.append("interrupted-task", {
			type: "run-start",
			at: "t1",
			runId: "original-run",
			model: "mock/task-resume",
		});
		const model = new MockLanguageModelV3({
			doStream: async () => ({
				stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
					{ type: "stream-start", warnings: [] },
					{
						type: "tool-call",
						toolCallId: "submit-resume",
						toolName: "submit_deliverable",
						input: JSON.stringify({ deliverable: { done: true } }),
					},
					finish(),
				]),
			}),
		});
		const agentic = createAgentic({ apiKey: "test", storage, getModel: () => model });

		await expect(
			agentic.task({
				id: "interrupted-task",
				agent: { model: "mock/task-resume" },
				prompt: "ignored because history exists",
				deliverable: z.object({ done: z.boolean() }),
			}),
		).resolves.toMatchObject({ status: "submitted", deliverable: { done: true } });

		const events = await storage.load("interrupted-task");
		expect(events.filter((event) => event.type === "run-start")).toHaveLength(1);
		expect(events).toContainEqual(
			expect.objectContaining({ type: "run-resume", runId: "original-run" }),
		);
		expect(events).toContainEqual(
			expect.objectContaining({
				type: "run-end",
				runId: "original-run",
				status: "completed",
			}),
		);
	});

	it("closes a persisted submitted outcome without another model call", async () => {
		const storage = memoryStorage();
		await storage.append("settled-submit", {
			type: "run-start",
			at: "t0",
			runId: "r-submit",
			model: "mock/settled-submit",
		});
		await storage.append("settled-submit", {
			type: "step",
			at: "t1",
			runId: "r-submit",
			messages: stored(
				{
					role: "assistant",
					content: [
						{
							type: "tool-call",
							toolCallId: "submit",
							toolName: "submit_deliverable",
							input: { deliverable: { value: "kept" } },
						},
					],
				},
				{
					role: "tool",
					content: [
						{
							type: "tool-result",
							toolCallId: "submit",
							toolName: "submit_deliverable",
							output: { type: "json", value: { accepted: true } },
						},
					],
				},
			),
			inputMessageIds: [],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "tool-calls",
			usage: usage(),
		});
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				throw new Error("model should not run");
			},
		});
		const options = {
			id: "settled-submit",
			agent: { model: "mock/settled-submit" },
			prompt: "ignored",
			deliverable: z.object({ value: z.string() }),
		};

		const failingRuntime = createAgentic({
			storage: {
				...storage,
				append: async (sessionId: string, event: StoredEvent) => {
					if (event.type === "run-end") throw new Error("reconciliation write failed");
					await storage.append(sessionId, event);
				},
			},
			getModel: () => model,
		});
		await expect(failingRuntime.task(options)).rejects.toThrow("reconciliation write failed");
		expect(replaySession(await storage.load(options.id)).interruptedRunId).toBe("r-submit");

		const recovered = createAgentic({ storage, getModel: () => model });
		await expect(recovered.task(options)).resolves.toMatchObject({
			status: "submitted",
			deliverable: { value: "kept" },
		});
		expect(calls).toBe(0);
		expect(replaySession(await storage.load(options.id)).interruptedRunId).toBeNull();
	});

	it("closes a persisted cancelled outcome without another model call", async () => {
		const storage = memoryStorage();
		await storage.append("settled-cancel", {
			type: "run-start",
			at: "t0",
			runId: "r-cancel",
			model: "mock/settled-cancel",
		});
		await storage.append("settled-cancel", {
			type: "step",
			at: "t1",
			runId: "r-cancel",
			messages: stored(
				{
					role: "assistant",
					content: [
						{
							type: "tool-call",
							toolCallId: "cancel",
							toolName: "cancel_task",
							input: { reason: "cannot continue" },
						},
					],
				},
				{
					role: "tool",
					content: [
						{
							type: "tool-result",
							toolCallId: "cancel",
							toolName: "cancel_task",
							output: { type: "json", value: { ok: true } },
						},
					],
				},
			),
			inputMessageIds: [],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "tool-calls",
			usage: usage(),
		});
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				throw new Error("model should not run");
			},
		});
		const agentic = createAgentic({ storage, getModel: () => model });

		await expect(
			agentic.task({
				id: "settled-cancel",
				agent: { model: "mock/settled-cancel" },
				prompt: "ignored",
			}),
		).resolves.toMatchObject({ status: "cancelled", reason: "cannot continue" });
		expect(calls).toBe(0);
		expect((await storage.load("settled-cancel")).at(-1)).toMatchObject({
			type: "run-end",
			status: "cancelled",
		});
		expect(replaySession(await storage.load("settled-cancel")).interruptedRunId).toBeNull();
	});
});

describe("ledger durability", () => {
	it("never reports completion when persisting a finished step fails", async () => {
		setContextWindow("mock/storage-failure", 100_000);
		const base = memoryStorage();
		const storage = {
			...base,
			append: async (sessionId: string, event: StoredEvent) => {
				if (event.type === "step") throw new Error("database unavailable");
				await base.append(sessionId, event);
			},
		};
		await storage.append("storage-failure", {
			type: "user-message",
			at: "t0",
			id: "u-1",
			message: { role: "user", content: "hello" },
		});
		const model = new MockLanguageModelV3({
			doStream: async () => ({
				stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
					{ type: "stream-start", warnings: [] },
					{ type: "text-start", id: "1" },
					{ type: "text-delta", id: "1", delta: "unpersisted answer" },
					{ type: "text-end", id: "1" },
					{
						type: "finish",
						finishReason: { unified: "stop", raw: "stop" },
						usage: {
							inputTokens: {
								total: 10,
								noCache: 10,
								cacheRead: undefined,
								cacheWrite: undefined,
							},
							outputTokens: { total: 5, text: 5, reasoning: undefined },
						},
					},
				]),
			}),
		});

		const result = await runLoop({
			sessionId: "storage-failure",
			agent: { model: "mock/storage-failure" },
			storage,
			getModel: () => model,
		});

		expect(result.status).toBe("failed");
		expect(result.error?.message).toContain("database unavailable");
		expect((await base.load("storage-failure")).some((event) => event.type === "run-end")).toBe(
			true,
		);
	});

	it("passes AgentConfig.toolChoice through to the model call", async () => {
		setContextWindow("mock/tool-choice", 100_000);
		const storage = memoryStorage();
		await storage.append("tool-choice", {
			type: "user-message",
			at: "t0",
			id: "u-1",
			message: { role: "user", content: "use a tool" },
		});
		let receivedToolChoice: unknown;
		const model = new MockLanguageModelV3({
			doStream: async ({ toolChoice }) => {
				receivedToolChoice = toolChoice;
				return {
					stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
						{ type: "stream-start", warnings: [] },
						{ type: "text-start", id: "1" },
						{ type: "text-delta", id: "1", delta: "done" },
						{ type: "text-end", id: "1" },
						{
							type: "finish",
							finishReason: { unified: "stop", raw: "stop" },
							usage: {
								inputTokens: {
									total: 1,
									noCache: 1,
									cacheRead: undefined,
									cacheWrite: undefined,
								},
								outputTokens: { total: 1, text: 1, reasoning: undefined },
							},
						},
					]),
				};
			},
		});
		const lookup = tool({ inputSchema: z.object({}), execute: async () => ({ ok: true }) });

		await runLoop({
			sessionId: "tool-choice",
			agent: { model: "mock/tool-choice", tools: { lookup }, toolChoice: "required" },
			storage,
			getModel: () => model,
		});
		expect(receivedToolChoice).toEqual({ type: "required" });
	});

	it("does not retry after visible partial output", async () => {
		setContextWindow("mock/visible-partial", 100_000);
		const storage = memoryStorage();
		await storage.append("visible-partial", {
			type: "user-message",
			at: "t0",
			id: "u-1",
			message: { role: "user", content: "answer once" },
		});
		let calls = 0;
		const deltas: string[] = [];
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				return {
					stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
						{ type: "stream-start", warnings: [] },
						{ type: "text-start", id: "1" },
						{ type: "text-delta", id: "1", delta: "partial" },
						{ type: "error", error: new Error("502 connection dropped") },
					]),
				};
			},
		});

		const result = await runLoop({
			sessionId: "visible-partial",
			agent: { model: "mock/visible-partial", retry: { maxAttempts: 3, baseDelayMs: 0 } },
			storage,
			getModel: () => model,
			onPart: (part) => {
				if (part.type === "text-delta") deltas.push(part.text);
			},
		});

		expect(result.status).toBe("failed");
		expect(calls).toBe(1);
		expect(deltas).toEqual(["partial"]);
	});

	it("serializes custom-provider step appends in invocation order", async () => {
		const modelId = "mock/serialized-custom-storage";
		setContextWindow(modelId, 100_000);
		const events: StoredEvent[] = [];
		let releaseFirstStep!: () => void;
		const firstStepGate = new Promise<void>((resolve) => {
			releaseFirstStep = resolve;
		});
		let markFirstStepStarted!: () => void;
		const firstStepStarted = new Promise<void>((resolve) => {
			markFirstStepStarted = resolve;
		});
		let stepAppends = 0;
		const storage = {
			async append(_sessionId: string, event: StoredEvent) {
				if (event.type === "step" && stepAppends++ === 0) {
					markFirstStepStarted();
					await firstStepGate;
				}
				events.push(event);
			},
			load: async () => [...events],
			listSessions: async () => ["serialized-custom-storage"],
		};
		let modelCall = 0;
		let markSecondModelCall!: () => void;
		const secondModelCall = new Promise<void>((resolve) => {
			markSecondModelCall = resolve;
		});
		const finishPart: LanguageModelV3StreamPart = {
			type: "finish",
			finishReason: { unified: "stop", raw: "stop" },
			usage: {
				inputTokens: { total: 1, noCache: 1, cacheRead: undefined, cacheWrite: undefined },
				outputTokens: { total: 1, text: 1, reasoning: undefined },
			},
		};
		const model = new MockLanguageModelV3({
			doStream: async () => {
				modelCall += 1;
				if (modelCall === 1) {
					return {
						stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
							{ type: "stream-start", warnings: [] },
							{
								type: "tool-call",
								toolCallId: "lookup-1",
								toolName: "lookup",
								input: "{}",
							},
							{
								...finishPart,
								finishReason: { unified: "tool-calls", raw: "tool_calls" },
							},
						]),
					};
				}
				markSecondModelCall();
				return {
					stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
						{ type: "stream-start", warnings: [] },
						{ type: "text-start", id: "1" },
						{ type: "text-delta", id: "1", delta: "done" },
						{ type: "text-end", id: "1" },
						finishPart,
					]),
				};
			},
		});
		const lookup = tool({ inputSchema: z.object({}), execute: async () => ({ ok: true }) });
		const session = createAgentic({ storage, getModel: () => model }).session(
			"serialized-custom-storage",
			{ model: modelId, tools: { lookup } },
		);

		const result = session.send("go");
		await Promise.all([firstStepStarted, secondModelCall]);
		releaseFirstStep();
		await expect(result).resolves.toMatchObject({ status: "completed", text: "done" });
		const steps = events.filter(
			(event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step",
		);
		expect(steps).toHaveLength(2);
		expect(steps[0].finishReason).toBe("tool-calls");
		expect(steps[1].finishReason).toBe("stop");
		expect(events.at(-1)?.type).toBe("run-end");
	});
});

describe("explicit cancellation durability", () => {
	const hangingTextStream = (
		abortSignal: AbortSignal | undefined,
		text?: string,
		onStarted?: () => void,
	) =>
		new ReadableStream<LanguageModelV3StreamPart>({
			start(controller) {
				const abort = () => controller.error(abortSignal?.reason ?? new Error("aborted"));
				abortSignal?.addEventListener("abort", abort, { once: true });
				controller.enqueue({ type: "stream-start", warnings: [] });
				if (text !== undefined) {
					controller.enqueue({ type: "text-start", id: "partial" });
					controller.enqueue({ type: "text-delta", id: "partial", delta: text });
				}
				onStarted?.();
			},
		});
	const completedTextStream = (text: string) =>
		convertArrayToReadableStream<LanguageModelV3StreamPart>([
			{ type: "stream-start", warnings: [] },
			{ type: "text-start", id: "complete" },
			{ type: "text-delta", id: "complete", delta: text },
			{ type: "text-end", id: "complete" },
			{
				type: "finish",
				finishReason: { unified: "stop", raw: "stop" },
				usage: {
					inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
					outputTokens: { total: 5, text: 5, reasoning: undefined },
				},
			},
		]);

	it("durably closes a send whose signal was already aborted", async () => {
		const modelId = "mock/cancel-pre-aborted";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const abort = new AbortController();
		abort.abort(new Error("already stopped"));
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				throw new Error("model must not run");
			},
		});
		const session = createAgentic({ storage, getModel: () => model }).session(
			"cancel-pre-aborted",
			{ model: modelId },
		);

		await expect(session.send("hello", { abortSignal: abort.signal })).resolves.toMatchObject({
			status: "cancelled",
			text: "",
			error: { message: "already stopped" },
		});
		expect(calls).toBe(0);
		const events = await storage.load("cancel-pre-aborted");
		expect(events.map((event) => event.type)).toEqual([
			"user-message",
			"run-start",
			"step",
			"run-end",
		]);
		expect(events.at(-2)).toMatchObject({
			type: "step",
			finishReason: "cancelled",
			interrupted: { status: "cancelled" },
		});
		expect(replaySession(events).pendingMessages).toBe(0);
	});

	it("persists and returns assistant text when aborted after the first delta", async () => {
		const modelId = "mock/cancel-after-delta";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const abort = new AbortController();
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => ({
				stream: hangingTextStream(abortSignal, "kept partial answer"),
			}),
		});
		const session = createAgentic({ storage, getModel: () => model }).session(
			"cancel-after-delta",
			{ model: modelId },
		);

		const result = await session.send("hello", {
			abortSignal: abort.signal,
			onPart: (part) => {
				if (part.type === "text-delta") abort.abort(new Error("user pressed stop"));
			},
		});

		expect(result).toMatchObject({
			status: "cancelled",
			text: "kept partial answer",
			error: { message: "user pressed stop" },
		});
		const events = await storage.load("cancel-after-delta");
		const step = events.find(
			(event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step",
		);
		expect(step).toMatchObject({
			finishReason: "cancelled",
			acknowledgesInput: true,
			interrupted: { status: "cancelled", error: { message: "user pressed stop" } },
		});
		expect(JSON.stringify(step?.messages)).toContain("kept partial answer");
		expect(events.at(-1)).toMatchObject({ type: "run-end", status: "cancelled" });
		expect(JSON.stringify(await session.messages())).toContain("kept partial answer");
		expect(replaySession(events).pendingMessages).toBe(0);
	});

	it("records terminal cancellation even when stopped before the first token", async () => {
		const modelId = "mock/cancel-before-token";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const abort = new AbortController();
		let markStarted!: () => void;
		const started = new Promise<void>((resolve) => {
			markStarted = resolve;
		});
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => ({
				stream: hangingTextStream(abortSignal, undefined, markStarted),
			}),
		});
		const session = createAgentic({ storage, getModel: () => model }).session(
			"cancel-before-token",
			{ model: modelId },
		);

		const pending = session.send("hello", { abortSignal: abort.signal });
		await started;
		abort.abort(new Error("stopped before output"));
		await expect(pending).resolves.toMatchObject({ status: "cancelled", text: "" });

		const events = await storage.load("cancel-before-token");
		const step = events.find(
			(event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step",
		);
		expect(step).toMatchObject({
			messages: [],
			finishReason: "cancelled",
			acknowledgesInput: true,
			interrupted: { status: "cancelled" },
		});
		expect(replaySession(events)).toMatchObject({
			interruptedRunId: null,
			pendingMessages: 0,
		});
	});

	it("does not acknowledge queued input when cancellation lands before streamText", async () => {
		const modelId = "mock/cancel-before-stream-text";
		setContextWindow(modelId, 100_000);
		const base = memoryStorage();
		const sessionId = "cancel-before-stream-text";
		await base.append(sessionId, {
			type: "user-message",
			at: "t0",
			id: "q-before-stream",
			message: { role: "user", content: "queued work" },
			meta: { queued: true, queueId: "q-before-stream" },
		});
		let releaseRunStart!: () => void;
		const runStartGate = new Promise<void>((resolve) => {
			releaseRunStart = resolve;
		});
		let markRunStart!: () => void;
		const runStartSeen = new Promise<void>((resolve) => {
			markRunStart = resolve;
		});
		const storage = {
			...base,
			append: async (id: string, event: StoredEvent) => {
				if (event.type === "run-start") {
					markRunStart();
					await runStartGate;
				}
				await base.append(id, event);
			},
		};
		const abort = new AbortController();
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				throw new Error("model must not run");
			},
		});
		const pending = runLoop({
			sessionId,
			agent: { model: modelId },
			storage,
			getModel: () => model,
			abortSignal: abort.signal,
		});

		await runStartSeen;
		abort.abort(new Error("stop before model entry"));
		releaseRunStart();
		await expect(pending).resolves.toMatchObject({ status: "cancelled" });

		const events = await base.load(sessionId);
		const cancellation = events.find(
			(event): event is Extract<StoredEvent, { type: "step" }> =>
				event.type === "step" && event.interrupted !== undefined,
		);
		expect(cancellation?.inputQueueIds).toEqual([]);
		expect(replaySession(events).pendingQueueIds).toEqual(["q-before-stream"]);
		expect(calls).toBe(0);
	});

	it("keeps a completed tool step and salvages only the partial current step", async () => {
		const modelId = "mock/cancel-after-completed-step";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const abort = new AbortController();
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => {
				calls += 1;
				if (calls === 1) {
					return {
						stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
							{ type: "stream-start", warnings: [] },
							{
								type: "tool-call",
								toolCallId: "lookup-complete",
								toolName: "lookup",
								input: JSON.stringify({ query: "answer" }),
							},
							{
								type: "finish",
								finishReason: { unified: "tool-calls", raw: "tool_calls" },
								usage: {
									inputTokens: {
										total: 10,
										noCache: 10,
										cacheRead: undefined,
										cacheWrite: undefined,
									},
									outputTokens: { total: 5, text: 0, reasoning: undefined },
								},
							},
						]),
					};
				}
				return { stream: hangingTextStream(abortSignal, "partial continuation") };
			},
		});
		const lookup = tool({
			inputSchema: z.object({ query: z.string() }),
			execute: async () => ({ value: 42 }),
		});
		const session = createAgentic({ storage, getModel: () => model }).session(
			"cancel-after-completed-step",
			{ model: modelId, tools: { lookup } },
		);

		const result = await session.send("look it up", {
			abortSignal: abort.signal,
			onPart: (part) => {
				if (part.type === "text-delta") abort.abort(new Error("stop continuation"));
			},
		});

		expect(result).toMatchObject({ status: "cancelled", text: "partial continuation" });
		const steps = (await storage.load("cancel-after-completed-step")).filter(
			(event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step",
		);
		expect(steps).toHaveLength(2);
		expect(steps[0]).toMatchObject({ finishReason: "tool-calls" });
		expect(steps[0].interrupted).toBeUndefined();
		expect(steps[1]).toMatchObject({
			finishReason: "cancelled",
			interrupted: { status: "cancelled" },
		});
		expect(JSON.stringify(steps[1].messages)).toContain("partial continuation");
		expect(JSON.stringify(steps[1].messages)).not.toContain("lookup-complete");
	});

	it("pairs an interrupted tool call with a synthetic replay-safe result", async () => {
		const modelId = "mock/cancel-tool-call";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const abort = new AbortController();
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => ({
				stream: new ReadableStream<LanguageModelV3StreamPart>({
					start(controller) {
						abortSignal?.addEventListener(
							"abort",
							() => controller.error(abortSignal.reason ?? new Error("aborted")),
							{ once: true },
						);
						controller.enqueue({ type: "stream-start", warnings: [] });
						controller.enqueue({
							type: "tool-call",
							toolCallId: "lookup-interrupted",
							toolName: "lookup",
							input: JSON.stringify({ query: "weather" }),
						});
					},
				}),
			}),
		});
		const lookup = tool({ inputSchema: z.object({ query: z.string() }) });
		const session = createAgentic({ storage, getModel: () => model }).session("cancel-tool-call", {
			model: modelId,
			tools: { lookup },
		});

		await expect(
			session.send("check", {
				abortSignal: abort.signal,
				onPart: (part) => {
					if (part.type === "tool-call") abort.abort(new Error("stop before execution"));
				},
			}),
		).resolves.toMatchObject({ status: "cancelled" });

		const events = await storage.load("cancel-tool-call");
		const step = events.find(
			(event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step",
		);
		expect(step?.messages.map((entry) => entry.message.role)).toEqual(["assistant", "tool"]);
		expect(step?.messages.every((entry) => typeof entry.id === "string")).toBe(true);
		expect(JSON.stringify(step?.messages)).toContain("lookup-interrupted");
		expect(JSON.stringify(step?.messages)).toContain("run was cancelled");
		const replayed = replaySession(events);
		expect(replayed.repaired).toBe(0);
		expect(replayed.messages.map((entry) => entry.message.role)).toEqual([
			"user",
			"assistant",
			"tool",
		]);
	});

	it("does not promote a preliminary tool result to a final result on cancellation", async () => {
		const modelId = "mock/cancel-preliminary-tool-result";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const abort = new AbortController();
		const model = new MockLanguageModelV3({
			doStream: async () => ({
				stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
					{ type: "stream-start", warnings: [] },
					{
						type: "tool-call",
						toolCallId: "streaming-tool",
						toolName: "streaming_lookup",
						input: JSON.stringify({ query: "weather" }),
					},
					{
						type: "finish",
						finishReason: { unified: "tool-calls", raw: "tool_calls" },
						usage: {
							inputTokens: {
								total: 10,
								noCache: 10,
								cacheRead: undefined,
								cacheWrite: undefined,
							},
							outputTokens: { total: 1, text: 0, reasoning: undefined },
						},
					},
				]),
			}),
		});
		const session = createAgentic({ storage, getModel: () => model }).session(
			"cancel-preliminary-tool-result",
			{
				model: modelId,
				tools: {
					streaming_lookup: tool({
						inputSchema: z.object({ query: z.string() }),
						execute: async function* (_input, { abortSignal }) {
							yield { stage: "preview-only" };
							await new Promise<never>((_resolve, reject) => {
								const stop = () => reject(abortSignal?.reason ?? new Error("aborted"));
								if (abortSignal?.aborted) stop();
								else abortSignal?.addEventListener("abort", stop, { once: true });
							});
						},
					}),
				},
			},
		);

		await expect(
			session.send("check", {
				abortSignal: abort.signal,
				onPart: (part) => {
					if (part.type === "tool-result" && part.preliminary) {
						abort.abort(new Error("stop during preview"));
					}
				},
			}),
		).resolves.toMatchObject({ status: "cancelled" });

		const events = await storage.load("cancel-preliminary-tool-result");
		const saved = JSON.stringify(events);
		expect(saved).not.toContain("preview-only");
		expect(saved).toContain("run was cancelled");
		expect(replaySession(events).repaired).toBe(0);
	});

	it("force-cancels a live run through the public runtime API", async () => {
		const modelId = "mock/runtime-cancel-live";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		let markDelta!: () => void;
		const deltaSeen = new Promise<void>((resolve) => {
			markDelta = resolve;
		});
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => ({
				stream: hangingTextStream(abortSignal, "kept through runtime cancel"),
			}),
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const session = agentic.session("runtime-cancel-live", { model: modelId });

		const pending = session.send("hello", {
			onPart: (part) => {
				if (part.type === "text-delta") markDelta();
			},
		});
		await deltaSeen;
		expect(agentic.isRunning("runtime-cancel-live")).toBe(true);
		expect(agentic.cancel("runtime-cancel-live", new Error("server stop"))).toBe(true);
		expect(agentic.cancel("runtime-cancel-live", new Error("duplicate stop"))).toBe(false);
		await expect(pending).resolves.toMatchObject({
			status: "cancelled",
			text: "kept through runtime cancel",
			error: { message: "server stop" },
		});
		expect(agentic.isRunning("runtime-cancel-live")).toBe(false);
	});

	it("starts queued work in a successor run that can be cancelled independently", async () => {
		const modelId = "mock/runtime-cancel-queued-successor";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => {
				calls += 1;
				return {
					stream: hangingTextStream(
						abortSignal,
						calls === 1 ? "partial answer to A" : "partial answer to B",
					),
				};
			},
		});
		let queued!: () => void;
		const queuedDurably = new Promise<void>((resolve) => {
			queued = resolve;
		});
		const agentic = createAgentic({
			storage,
			getModel: () => model,
			onEvent: (event) => {
				if (event.type === "queued-message") queued();
			},
		});
		const session = agentic.session("runtime-cancel-queued-successor", { model: modelId });
		let firstDelta!: () => void;
		const firstStreaming = new Promise<void>((resolve) => {
			firstDelta = resolve;
		});
		let secondDelta!: () => void;
		const secondStreaming = new Promise<void>((resolve) => {
			secondDelta = resolve;
		});

		const first = session.send("A", {
			onPart: (part) => {
				if (part.type === "text-delta") firstDelta();
			},
		});
		await firstStreaming;
		const second = session.send("B", {
			onPart: (part) => {
				if (part.type === "text-delta") secondDelta();
			},
		});
		await queuedDurably;

		expect(agentic.cancel(session.id, new Error("stop A"))).toBe(true);
		await expect(first).resolves.toMatchObject({
			status: "cancelled",
			text: "partial answer to A",
			error: { message: "stop A" },
		});
		await secondStreaming;
		expect(calls).toBe(2);
		expect(agentic.isRunning(session.id)).toBe(true);
		expect(agentic.cancel(session.id, new Error("stop B"))).toBe(true);
		await expect(second).resolves.toMatchObject({
			status: "cancelled",
			text: "partial answer to B",
			error: { message: "stop B" },
		});

		const events = await storage.load(session.id);
		expect(events.filter((event) => event.type === "run-start")).toHaveLength(2);
		const steps = events.filter(
			(event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step",
		);
		expect(steps[0].inputQueueIds).toEqual([]);
		expect(steps[1].inputQueueIds).toHaveLength(1);
		expect(events.filter((event) => event.type === "run-end")).toEqual([
			expect.objectContaining({
				status: "cancelled",
				error: expect.objectContaining({ message: "stop A" }),
			}),
			expect.objectContaining({
				status: "cancelled",
				error: expect.objectContaining({ message: "stop B" }),
			}),
		]);
		expect(replaySession(events).pendingMessages).toBe(0);
	});

	it("does not replay queued input already included in the cancelled model pass", async () => {
		const modelId = "mock/runtime-cancel-queued-included";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		let releaseFirst!: () => void;
		const firstGate = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});
		let markFirstStarted!: () => void;
		const firstStarted = new Promise<void>((resolve) => {
			markFirstStarted = resolve;
		});
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => {
				calls += 1;
				if (calls === 1) {
					markFirstStarted();
					await firstGate;
					return { stream: completedTextStream("answer to A") };
				}
				return { stream: hangingTextStream(abortSignal, "partial answer including B") };
			},
		});
		let queued!: () => void;
		const queuedDurably = new Promise<void>((resolve) => {
			queued = resolve;
		});
		const agentic = createAgentic({
			storage,
			getModel: () => model,
			onEvent: (event) => {
				if (event.type === "queued-message") queued();
			},
		});
		const session = agentic.session("runtime-cancel-queued-included", { model: modelId });
		let secondDelta!: () => void;
		const secondStreaming = new Promise<void>((resolve) => {
			secondDelta = resolve;
		});

		const first = session.send("A");
		await firstStarted;
		const second = session.send("B", {
			onPart: (part) => {
				if (part.type === "text-delta") secondDelta();
			},
		});
		await queuedDurably;
		releaseFirst();
		await secondStreaming;
		expect(agentic.cancel(session.id, new Error("stop pass containing B"))).toBe(true);

		await expect(Promise.all([first, second])).resolves.toEqual([
			expect.objectContaining({ status: "cancelled", text: "partial answer including B" }),
			expect.objectContaining({ status: "cancelled", text: "partial answer including B" }),
		]);
		expect(calls).toBe(2);
		const events = await storage.load(session.id);
		expect(events.filter((event) => event.type === "run-start")).toHaveLength(1);
		expect(events.filter((event) => event.type === "run-end")).toHaveLength(1);
		const steps = events.filter(
			(event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step",
		);
		expect(steps[1].inputQueueIds).toHaveLength(1);
		expect(replaySession(events).pendingMessages).toBe(0);
	});

	it("batches every queued message into exactly one successor after cancellation", async () => {
		const modelId = "mock/runtime-cancel-queued-batch";
		setContextWindow(modelId, 100_000);
		const base = memoryStorage();
		let releaseQueuedAppend!: () => void;
		const queuedAppendGate = new Promise<void>((resolve) => {
			releaseQueuedAppend = resolve;
		});
		let markQueuedAppendStarted!: () => void;
		const queuedAppendStarted = new Promise<void>((resolve) => {
			markQueuedAppendStarted = resolve;
		});
		let blockFirstQueuedAppend = true;
		const storage = {
			...base,
			append: async (sessionId: string, event: StoredEvent) => {
				if (
					blockFirstQueuedAppend &&
					event.type === "user-message" &&
					event.meta?.queued === true
				) {
					blockFirstQueuedAppend = false;
					markQueuedAppendStarted();
					await queuedAppendGate;
				}
				await base.append(sessionId, event);
			},
		};
		const prompts: ModelMessage[][] = [];
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal, prompt }) => {
				calls += 1;
				prompts.push(prompt);
				if (calls === 1) {
					return { stream: hangingTextStream(abortSignal, "partial answer to A") };
				}
				return { stream: completedTextStream("one answer for B, C, and D") };
			},
		});
		const agentic = createAgentic({
			storage,
			getModel: () => model,
		});
		const session = agentic.session("runtime-cancel-queued-batch", { model: modelId });
		let firstDelta!: () => void;
		const firstStreaming = new Promise<void>((resolve) => {
			firstDelta = resolve;
		});

		const first = session.send("A", {
			onPart: (part) => {
				if (part.type === "text-delta") firstDelta();
			},
		});
		await firstStreaming;
		const queued = [session.send("B"), session.send("C"), session.send("D")];
		// B is blocked inside persistence while C and D are attached behind it.
		// Cancellation must wait for the complete intake set before any sender
		// takes the recovery snapshot.
		await queuedAppendStarted;

		expect(agentic.cancel(session.id, new Error("stop A only"))).toBe(true);
		releaseQueuedAppend();
		await expect(first).resolves.toMatchObject({
			status: "cancelled",
			text: "partial answer to A",
		});
		await expect(Promise.all(queued)).resolves.toEqual([
			expect.objectContaining({ status: "completed", text: "one answer for B, C, and D" }),
			expect.objectContaining({ status: "completed", text: "one answer for B, C, and D" }),
			expect.objectContaining({ status: "completed", text: "one answer for B, C, and D" }),
		]);

		expect(calls).toBe(2);
		const successorPrompt = JSON.stringify(prompts[1]);
		for (const content of ["B", "C", "D"]) {
			expect(successorPrompt.match(new RegExp(`"${content}"`, "g"))).toHaveLength(1);
		}
		const events = await storage.load(session.id);
		expect(events.filter((event) => event.type === "run-start")).toHaveLength(2);
		expect(events.filter((event) => event.type === "run-end")).toHaveLength(2);
		const steps = events.filter(
			(event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step",
		);
		expect(steps[0].inputQueueIds).toEqual([]);
		expect(steps[1].inputQueueIds).toHaveLength(3);
		expect(replaySession(events).pendingMessages).toBe(0);
	});

	it("can cancel the registered run before its initial durable append settles", async () => {
		const modelId = "mock/runtime-cancel-before-start";
		setContextWindow(modelId, 100_000);
		const base = memoryStorage();
		let releaseAppend!: () => void;
		const appendGate = new Promise<void>((resolve) => {
			releaseAppend = resolve;
		});
		let markAppendStarted!: () => void;
		const appendStarted = new Promise<void>((resolve) => {
			markAppendStarted = resolve;
		});
		let blockInput = true;
		const storage = {
			...base,
			append: async (sessionId: string, event: StoredEvent) => {
				if (blockInput && event.type === "user-message") {
					blockInput = false;
					markAppendStarted();
					await appendGate;
				}
				await base.append(sessionId, event);
			},
		};
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				throw new Error("model must not run");
			},
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const pending = agentic
			.session("runtime-cancel-before-start", { model: modelId })
			.send("hello");

		await appendStarted;
		expect(agentic.isRunning("runtime-cancel-before-start")).toBe(true);
		expect(agentic.cancel("runtime-cancel-before-start", new Error("early stop"))).toBe(true);
		releaseAppend();
		await expect(pending).resolves.toMatchObject({
			status: "cancelled",
			error: { message: "early stop" },
		});
		expect(calls).toBe(0);
		expect(agentic.isRunning("runtime-cancel-before-start")).toBe(false);
	});

	it("reconciles a saved cancellation marker without another model call", async () => {
		const modelId = "mock/cancel-run-end-crash";
		setContextWindow(modelId, 100_000);
		const base = memoryStorage();
		let rejectCancelledEnd = true;
		const flakyStorage = {
			...base,
			append: async (sessionId: string, event: StoredEvent) => {
				if (event.type === "run-end" && event.status === "cancelled" && rejectCancelledEnd) {
					rejectCancelledEnd = false;
					throw new Error("run-end unavailable");
				}
				await base.append(sessionId, event);
			},
		};
		const abort = new AbortController();
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => {
				calls += 1;
				return { stream: hangingTextStream(abortSignal, "durable before crash") };
			},
		});
		const first = createAgentic({ storage: flakyStorage, getModel: () => model }).session(
			"cancel-run-end-crash",
			{ model: modelId },
		);

		await expect(
			first.send("hello", {
				abortSignal: abort.signal,
				onPart: (part) => {
					if (part.type === "text-delta") abort.abort(new Error("intentional stop"));
				},
			}),
		).rejects.toThrow("run-end unavailable");
		const crashed = replaySession(await base.load("cancel-run-end-crash"));
		expect(crashed.interruptedRunId).not.toBeNull();
		expect(JSON.stringify(crashed.messages)).toContain("durable before crash");

		const recovered = createAgentic({ storage: base, getModel: () => model }).session(
			"cancel-run-end-crash",
			{ model: modelId },
		);
		await expect(recovered.resume()).resolves.toMatchObject({
			status: "cancelled",
			text: "durable before crash",
			error: { message: "intentional stop" },
		});
		expect(calls).toBe(1);
		expect((await base.load("cancel-run-end-crash")).at(-1)).toMatchObject({
			type: "run-end",
			status: "cancelled",
		});
		expect(await recovered.isInterrupted()).toBe(false);
	});

	it("auto-resume finalizes cancellation instead of re-entering the model", async () => {
		const modelId = "mock/cancel-auto-resume";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		await storage.append("cancel-auto-resume", {
			type: "user-message",
			at: "t0",
			id: "u-1",
			message: { role: "user", content: "hello" },
		});
		await storage.append("cancel-auto-resume", {
			type: "run-start",
			at: "t1",
			runId: "cancelled-run",
			model: modelId,
		});
		await storage.append("cancel-auto-resume", {
			type: "step",
			at: "t2",
			runId: "cancelled-run",
			messages: stored({ role: "assistant", content: "kept" }),
			inputMessageIds: ["u-1"],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "cancelled",
			usage: usage({
				inputTokens: null,
				outputTokens: null,
				totalTokens: null,
				cost: null,
				billedCost: null,
			}),
			interrupted: {
				status: "cancelled",
				error: { message: "stopped" },
			},
		});
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				throw new Error("model must not run");
			},
		});
		const agentic = createAgentic({
			storage,
			getModel: () => model,
			autoResume: () => ({ model: modelId }),
		});

		await agentic.resumeInterrupted();
		expect(calls).toBe(0);
		const events = await storage.load("cancel-auto-resume");
		expect(events.filter((event) => event.type === "run-resume")).toHaveLength(0);
		expect(events.filter((event) => event.type === "run-end")).toEqual([
			expect.objectContaining({ runId: "cancelled-run", status: "cancelled" }),
		]);
		expect(await agentic.interruptedSessions()).toEqual([]);
	});

	it("a resumed task reports a saved explicit cancellation without model work", async () => {
		const modelId = "mock/cancel-resumed-task";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		await storage.append("cancel-resumed-task", {
			type: "user-message",
			at: "t0",
			id: "u-1",
			message: { role: "user", content: "do work" },
		});
		await storage.append("cancel-resumed-task", {
			type: "run-start",
			at: "t1",
			runId: "cancelled-task-run",
			model: modelId,
		});
		await storage.append("cancel-resumed-task", {
			type: "step",
			at: "t2",
			runId: "cancelled-task-run",
			messages: stored({ role: "assistant", content: "partial task work" }),
			inputMessageIds: ["u-1"],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "cancelled",
			usage: usage(),
			interrupted: { status: "cancelled", error: { message: "operator stopped task" } },
		});
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				throw new Error("model must not run");
			},
		});
		const agentic = createAgentic({ storage, getModel: () => model });

		await expect(
			agentic.task({
				id: "cancel-resumed-task",
				agent: { model: modelId },
				prompt: "must not be appended",
			}),
		).resolves.toMatchObject({
			status: "cancelled",
			reason: "operator stopped task",
		});
		expect(calls).toBe(0);
		expect((await storage.load("cancel-resumed-task")).at(-1)).toMatchObject({
			type: "run-end",
			status: "cancelled",
		});
	});

	it("replays a normally closed explicit task cancellation without model work", async () => {
		const modelId = "mock/cancel-closed-task";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const abort = new AbortController();
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => {
				calls += 1;
				return { stream: hangingTextStream(abortSignal, "partial task work") };
			},
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const options = {
			id: "cancel-closed-task",
			agent: { model: modelId },
			prompt: "do work",
		};

		await expect(
			agentic.task({
				...options,
				abortSignal: abort.signal,
				onPart: (part) => {
					if (part.type === "text-delta") abort.abort(new Error("operator stopped task"));
				},
			}),
		).resolves.toMatchObject({ status: "cancelled", reason: "operator stopped task" });
		expect(replaySession(await storage.load(options.id)).interruptedRunId).toBeNull();

		await expect(agentic.task(options)).resolves.toMatchObject({
			status: "cancelled",
			reason: "operator stopped task",
		});
		expect(calls).toBe(1);
	});

	it("fails terminally when the cancellation step cannot be persisted", async () => {
		const modelId = "mock/cancel-step-storage-failure";
		setContextWindow(modelId, 100_000);
		const base = memoryStorage();
		const storage = {
			...base,
			append: async (sessionId: string, event: StoredEvent) => {
				if (event.type === "step" && event.interrupted) {
					throw new Error("cannot save partial response");
				}
				await base.append(sessionId, event);
			},
		};
		const abort = new AbortController();
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => ({
				stream: hangingTextStream(abortSignal, "not durable"),
			}),
		});
		const session = createAgentic({ storage, getModel: () => model }).session(
			"cancel-step-storage-failure",
			{ model: modelId },
		);

		await expect(
			session.send("hello", {
				abortSignal: abort.signal,
				onPart: (part) => {
					if (part.type === "text-delta") abort.abort(new Error("stop"));
				},
			}),
		).resolves.toMatchObject({
			status: "failed",
			text: "not durable",
			error: { message: "cannot save partial response" },
		});
		const events = await base.load("cancel-step-storage-failure");
		expect(events.some((event) => event.type === "step" && event.interrupted)).toBe(false);
		expect(events.at(-1)).toMatchObject({ type: "run-end", status: "failed" });
		expect(replaySession(events).interruptedRunId).toBeNull();
	});
});

describe("compaction with pending inputs", () => {
	const generatedSummary = {
		content: [{ type: "text" as const, text: "summary of earlier history" }],
		finishReason: { unified: "stop" as const, raw: "stop" },
		usage: {
			inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
			outputTokens: { total: 5, text: 5, reasoning: undefined },
		},
		warnings: [],
	};
	const answerStream = () =>
		convertArrayToReadableStream<LanguageModelV3StreamPart>([
			{ type: "stream-start", warnings: [] },
			{ type: "text-start", id: "1" },
			{ type: "text-delta", id: "1", delta: "current answer" },
			{ type: "text-end", id: "1" },
			{
				type: "finish",
				finishReason: { unified: "stop", raw: "stop" },
				usage: {
					inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
					outputTokens: { total: 5, text: 5, reasoning: undefined },
				},
			},
		]);

	it.each([
		["ordinary", false],
		["queued", true],
	] as const)("preserves one %s pending input across threshold compaction", async (_label, queued) => {
		const modelId = `mock/compact-pending-${queued ? "queued" : "ordinary"}`;
		setContextWindow(modelId, 100);
		const storage = memoryStorage();
		const sessionId = `compact-pending-${queued ? "queued" : "ordinary"}`;
		await storage.append(sessionId, {
			type: "user-message",
			at: "t0",
			id: "u-old",
			message: { role: "user", content: "old question" },
		});
		await storage.append(sessionId, {
			type: "step",
			at: "t1",
			runId: "old",
			messages: stored({ role: "assistant", content: "old answer" }),
			inputMessageIds: ["u-old"],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "stop",
			usage: usage({ inputTokens: 60, outputTokens: 20, totalTokens: 80 }),
		});
		await storage.append(sessionId, {
			type: "run-end",
			at: "t2",
			runId: "old",
			status: "completed",
		});
		await storage.append(sessionId, {
			type: "user-message",
			at: "t3",
			id: queued ? "q-current" : "u-current",
			message: { role: "user", content: "current question" },
			...(queued ? { meta: { queued: true, queueId: "q-current" } } : {}),
		});
		const model = new MockLanguageModelV3({
			doGenerate: async () => generatedSummary,
			doStream: async () => ({ stream: answerStream() }),
		});

		await expect(
			runLoop({
				sessionId,
				agent: { model: modelId, compaction: { limit: 0.5, keepRecent: 0 } },
				storage,
				getModel: () => model,
			}),
		).resolves.toMatchObject({ status: "completed", text: "current answer" });

		const events = await storage.load(sessionId);
		const compaction = events.find(
			(event): event is Extract<StoredEvent, { type: "compaction" }> => event.type === "compaction",
		);
		// The pending input keeps its ledger id in the rebased base, so replay
		// re-links it by id.
		expect(
			compaction?.messages.find((entry) =>
				JSON.stringify(entry.message.content).includes("current question"),
			)?.id,
		).toBe(queued ? "q-current" : "u-current");
		const replayed = replaySession(events);
		expect(replayed.pendingMessages).toBe(0);
		expect(
			replayed.messages.filter((entry) =>
				JSON.stringify(entry.message.content).includes("current question"),
			),
		).toHaveLength(1);
		const finalStep = [...events]
			.reverse()
			.find((event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step");
		expect(finalStep?.inputQueueIds).toEqual(queued ? ["q-current"] : []);
	});

	it("preserves an ordinary pending input across forced overflow compaction", async () => {
		const modelId = "mock/compact-pending-forced";
		setContextWindow(modelId, 100);
		const storage = memoryStorage();
		const sessionId = "compact-pending-forced";
		await storage.append(sessionId, {
			type: "user-message",
			at: "t0",
			id: "u-old",
			message: { role: "user", content: "old question" },
		});
		await storage.append(sessionId, {
			type: "step",
			at: "t1",
			runId: "old",
			messages: stored({ role: "assistant", content: "old answer" }),
			inputMessageIds: ["u-old"],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "stop",
			usage: usage({ inputTokens: 10, outputTokens: 5, totalTokens: 15 }),
		});
		await storage.append(sessionId, {
			type: "run-end",
			at: "t2",
			runId: "old",
			status: "completed",
		});
		await storage.append(sessionId, {
			type: "user-message",
			at: "t3",
			id: "u-current",
			message: { role: "user", content: "current forced question" },
		});
		let streamCall = 0;
		const model = new MockLanguageModelV3({
			doGenerate: async () => generatedSummary,
			doStream: async () => {
				streamCall += 1;
				if (streamCall === 1) {
					return {
						stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
							{ type: "stream-start", warnings: [] },
							{
								type: "error",
								error: new Error("maximum context length exceeded"),
							},
						]),
					};
				}
				return { stream: answerStream() };
			},
		});

		await expect(
			runLoop({
				sessionId,
				agent: { model: modelId, compaction: { limit: 0, keepRecent: 0 } },
				storage,
				getModel: () => model,
			}),
		).resolves.toMatchObject({ status: "completed", text: "current answer" });
		expect(streamCall).toBe(2);
		const replayed = replaySession(await storage.load(sessionId));
		expect(
			replayed.messages.filter((entry) =>
				JSON.stringify(entry.message.content).includes("current forced question"),
			),
		).toHaveLength(1);
	});

	it("does not acknowledge queued input when cancelled during pre-pass compaction", async () => {
		const modelId = "mock/compact-cancel-before-pass";
		setContextWindow(modelId, 100);
		const storage = memoryStorage();
		const sessionId = "compact-cancel-before-pass";
		await storage.append(sessionId, {
			type: "user-message",
			at: "t0",
			id: "u-old",
			message: { role: "user", content: "old question" },
		});
		await storage.append(sessionId, {
			type: "step",
			at: "t1",
			runId: "old",
			messages: stored({ role: "assistant", content: "old answer" }),
			inputMessageIds: ["u-old"],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "stop",
			usage: usage({ inputTokens: 60, outputTokens: 20, totalTokens: 80 }),
		});
		await storage.append(sessionId, {
			type: "run-end",
			at: "t2",
			runId: "old",
			status: "completed",
		});
		await storage.append(sessionId, {
			type: "user-message",
			at: "t3",
			id: "q-cancelled-compaction",
			message: { role: "user", content: "queued question" },
			meta: { queued: true, queueId: "q-cancelled-compaction" },
		});
		const abort = new AbortController();
		let markCompactionStarted!: () => void;
		const compactionStarted = new Promise<void>((resolve) => {
			markCompactionStarted = resolve;
		});
		let streamCalls = 0;
		const model = new MockLanguageModelV3({
			doGenerate: async ({ abortSignal }) => {
				markCompactionStarted();
				return new Promise<never>((_resolve, reject) => {
					const stop = () => reject(abortSignal?.reason ?? new Error("aborted"));
					if (abortSignal?.aborted) stop();
					else abortSignal?.addEventListener("abort", stop, { once: true });
				});
			},
			doStream: async () => {
				streamCalls += 1;
				return { stream: answerStream() };
			},
		});

		const pending = runLoop({
			sessionId,
			agent: { model: modelId, compaction: { limit: 0.5, keepRecent: 0 } },
			storage,
			getModel: () => model,
			abortSignal: abort.signal,
		});
		await compactionStarted;
		abort.abort(new Error("stop during compaction"));
		await expect(pending).resolves.toMatchObject({
			status: "cancelled",
			error: { message: "stop during compaction" },
		});

		const events = await storage.load(sessionId);
		const cancellation = events.find(
			(event): event is Extract<StoredEvent, { type: "step" }> =>
				event.type === "step" && event.interrupted !== undefined,
		);
		expect(cancellation).toMatchObject({
			inputQueueIds: [],
			acknowledgesInput: true,
			messages: [],
		});
		expect(streamCalls).toBe(0);
		expect(replaySession(events).pendingQueueIds).toEqual(["q-cancelled-compaction"]);
	});
});

describe("message queueing", () => {
	const textStream = (text: string) =>
		convertArrayToReadableStream<LanguageModelV3StreamPart>([
			{ type: "stream-start", warnings: [] },
			{ type: "text-start", id: "1" },
			{ type: "text-delta", id: "1", delta: text },
			{ type: "text-end", id: "1" },
			{
				type: "finish",
				finishReason: { unified: "stop", raw: "stop" },
				usage: {
					inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
					outputTokens: { total: 5, text: 5, reasoning: undefined },
				},
			},
		]);

	it("mailbox refuses enqueues once the run commits to ending", () => {
		const mailbox = createMailbox();
		expect(mailbox.tryEnqueue("q-1")).toBe(true);
		expect(mailbox.queued).toEqual(["q-1"]);
		mailbox.accepting = false;
		expect(mailbox.tryEnqueue("q-2")).toBe(false);
		expect(mailbox.queued).toEqual(["q-1"]);
	});

	it("durably queues a same-tick second send before the first run registers", async () => {
		const modelId = "mock/queue-same-tick";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const prompts: ModelMessage[][] = [];
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async ({ prompt }) => {
				calls += 1;
				prompts.push(prompt as ModelMessage[]);
				return { stream: textStream("answer to A and B") };
			},
		});
		const session = createAgentic({ storage, getModel: () => model }).session("queue-same-tick", {
			model: modelId,
		});

		const acceptances: Array<{ send: string; messageId: string; runId: string; queued: boolean }> =
			[];
		const partsAfterAccept: Record<"A" | "B", boolean[]> = { A: [], B: [] };
		const first = session.send("A", {
			onAccepted: (info) => acceptances.push({ send: "A", ...info }),
			onPart: () => partsAfterAccept.A.push(acceptances.some((entry) => entry.send === "A")),
		});
		const second = session.send("B", {
			onAccepted: (info) => acceptances.push({ send: "B", ...info }),
			onPart: () => partsAfterAccept.B.push(acceptances.some((entry) => entry.send === "B")),
		});
		const [firstResult, secondResult] = await Promise.all([first, second]);
		expect(firstResult).toMatchObject({ status: "completed", text: "answer to A and B" });
		expect(secondResult).toMatchObject({ status: "completed", text: "answer to A and B" });

		expect(calls).toBe(1);
		const users = prompts[0].filter((message) => message.role === "user");
		expect(users).toHaveLength(2);
		expect(JSON.stringify(users[0].content)).toContain("A");
		expect(JSON.stringify(users[1].content)).toContain("B");
		const events = await storage.load("queue-same-tick");
		expect(events.slice(0, 2).map((event) => event.type)).toEqual(["user-message", "user-message"]);
		expect(events.filter((event) => event.type === "run-start")).toHaveLength(1);
		expect(replaySession(events).pendingMessages).toBe(0);

		// Each same-tick send was accepted exactly once with its own durable
		// message id and the shared run's real id, before any of its parts.
		const userEvents = events.filter(
			(event): event is Extract<StoredEvent, { type: "user-message" }> =>
				event.type === "user-message",
		);
		expect(acceptances).toHaveLength(2);
		expect(acceptances.find((entry) => entry.send === "A")).toMatchObject({
			messageId: userEvents[0].id,
			runId: firstResult.runId,
			queued: false,
		});
		expect(acceptances.find((entry) => entry.send === "B")).toMatchObject({
			messageId: userEvents[1].id,
			runId: firstResult.runId,
			queued: true,
		});
		expect(firstResult.messageId).toBe(userEvents[0].id);
		expect(secondResult.messageId).toBe(userEvents[1].id);
		expect(secondResult.runId).toBe(firstResult.runId);
		expect(partsAfterAccept.A.length).toBeGreaterThan(0);
		expect(partsAfterAccept.B.length).toBeGreaterThan(0);
		expect(partsAfterAccept.A.every(Boolean)).toBe(true);
		expect(partsAfterAccept.B.every(Boolean)).toBe(true);
	});

	it("does not run a same-tick send twice behind an idle resume reservation", async () => {
		const modelId = "mock/resume-send-reservation";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		let calls = 0;
		let touches = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				if (calls % 2 === 1) {
					return {
						stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
							{ type: "stream-start", warnings: [] },
							{
								type: "tool-call",
								toolCallId: `touch-${calls}`,
								toolName: "touch",
								input: "{}",
							},
							{
								type: "finish",
								finishReason: { unified: "tool-calls", raw: "tool_calls" },
								usage: {
									inputTokens: {
										total: 10,
										noCache: 10,
										cacheRead: undefined,
										cacheWrite: undefined,
									},
									outputTokens: { total: 1, text: 0, reasoning: undefined },
								},
							},
						]),
					};
				}
				return { stream: textStream("done once") };
			},
		});
		const session = createAgentic({ storage, getModel: () => model }).session(
			"resume-send-reservation",
			{
				model: modelId,
				tools: {
					touch: tool({
						inputSchema: z.object({}),
						execute: async () => {
							touches += 1;
							return "touched";
						},
					}),
				},
			},
		);

		const resumed = session.resume();
		const sent = session.send("B");

		await expect(resumed).resolves.toBeNull();
		await expect(sent).resolves.toMatchObject({ status: "completed", text: "done once" });
		expect(touches).toBe(1);
		expect(calls).toBe(2);
		const events = await storage.load("resume-send-reservation");
		expect(events.filter((event) => event.type === "run-start")).toHaveLength(1);
		expect(events.filter((event) => event.type === "run-end")).toHaveLength(1);
	});

	it("does not expose a default send to an earlier same-tick queue:false run", async () => {
		const modelId = "mock/serialized-send-reservation";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const prompts: ModelMessage[][] = [];
		const model = new MockLanguageModelV3({
			doStream: async ({ prompt }) => {
				prompts.push(prompt as ModelMessage[]);
				return { stream: textStream(`answer-${prompts.length}`) };
			},
		});
		const session = createAgentic({ storage, getModel: () => model }).session(
			"serialized-send-reservation",
			{ model: modelId },
		);

		const dedicated = session.send("D", { queue: false });
		const next = session.send("B");

		await expect(Promise.all([dedicated, next])).resolves.toEqual([
			expect.objectContaining({ status: "completed", text: "answer-1" }),
			expect.objectContaining({ status: "completed", text: "answer-2" }),
		]);
		expect(prompts).toHaveLength(2);
		expect(JSON.stringify(prompts[0])).toContain("D");
		expect(JSON.stringify(prompts[0])).not.toContain("B");
		expect(JSON.stringify(prompts[1])).toContain("B");
		const events = await storage.load("serialized-send-reservation");
		const firstRunEnd = events.findIndex((event) => event.type === "run-end");
		const bInput = events.findIndex(
			(event) =>
				event.type === "user-message" && JSON.stringify(event.message.content).includes("B"),
		);
		expect(firstRunEnd).toBeGreaterThanOrEqual(0);
		expect(bInput).toBeGreaterThan(firstRunEnd);
	});

	it("runLoop answers a message queued mid-run, hoisted after model output", async () => {
		setContextWindow("mock/model", 100_000);
		const storage = memoryStorage();
		const sessionId = "queue-live";
		await storage.append(sessionId, {
			type: "user-message",
			at: "t0",
			id: "u-first",
			message: { role: "user", content: "first question" },
		});

		const mailbox = createMailbox();
		const prompts: Array<Array<{ role: string }>> = [];
		let call = 0;
		const model = new MockLanguageModelV3({
			doStream: async ({ prompt }) => {
				prompts.push(prompt as Array<{ role: string }>);
				call += 1;
				if (call === 1) {
					// a second send() lands while the first answer is streaming:
					// ledger append first (durable), then the mailbox signal
					await storage.append(sessionId, {
						type: "user-message",
						at: "t1",
						id: "q-1",
						message: { role: "user", content: "second question" },
						meta: { queued: true, queueId: "q-1" },
					});
					mailbox.tryEnqueue("q-1");
				}
				return {
					stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
						{ type: "stream-start", warnings: [] },
						{ type: "text-start", id: "1" },
						{ type: "text-delta", id: "1", delta: call === 1 ? "answer one" : "answer two" },
						{ type: "text-end", id: "1" },
						{
							type: "finish",
							finishReason: { unified: "stop", raw: "stop" },
							usage: {
								inputTokens: {
									total: 10,
									noCache: 10,
									cacheRead: undefined,
									cacheWrite: undefined,
								},
								outputTokens: { total: 5, text: 5, reasoning: undefined },
							},
						},
					]),
				};
			},
		});

		const result = await runLoop({
			sessionId,
			agent: { model: "mock/model" },
			storage,
			getModel: () => model,
			mailbox,
		});

		expect(result.status).toBe("completed");
		expect(result.text).toBe("answer two");
		// the run did NOT end after answering the first question
		expect(prompts).toHaveLength(2);
		// the queued message is the newest input in the second request, even
		// though it sits before the first answer in the ledger
		const lastMessage = prompts[1][prompts[1].length - 1];
		expect(lastMessage.role).toBe("user");
		// one run, two steps, closed cleanly; mailbox fully consumed
		const events = await storage.load(sessionId);
		expect(events.filter((e) => e.type === "step")).toHaveLength(2);
		expect(events.filter((e) => e.type === "run-start")).toHaveLength(1);
		expect(events.filter((e) => e.type === "run-end")).toHaveLength(1);
		expect(mailbox.queued).toHaveLength(0);
		expect(mailbox.accepting).toBe(false);
		// and the whole thing replays with nothing pending
		const finalReplay = replaySession(events);
		expect(finalReplay.pendingMessages).toBe(0);
		expect(finalReplay.messages.map((entry) => entry.message.role)).toEqual([
			"user",
			"assistant",
			"user",
			"assistant",
		]);
	});

	it("re-drives an accepted queued send when the live run fails before pickup", async () => {
		const modelId = "mock/queue-fail-before-pickup";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		let releaseFirst!: () => void;
		const firstGate = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});
		let markFirstStarted!: () => void;
		const firstStarted = new Promise<void>((resolve) => {
			markFirstStarted = resolve;
		});
		let markQueued!: () => void;
		const queued = new Promise<void>((resolve) => {
			markQueued = resolve;
		});
		const prompts: ModelMessage[][] = [];
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async ({ prompt }) => {
				calls += 1;
				prompts.push(prompt as ModelMessage[]);
				if (calls === 1) {
					markFirstStarted();
					await firstGate;
					return {
						stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
							{ type: "stream-start", warnings: [] },
							{ type: "error", error: new Error("Invalid API key") },
						]),
					};
				}
				return { stream: textStream("answer to B") };
			},
		});
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
			onEvent: (event) => {
				if (event.type === "queued-message") markQueued();
			},
		});
		const session = agentic.session("queue-fail-before-pickup", { model: modelId });

		const first = session.send("A");
		await firstStarted;
		const second = session.send("B");
		await queued;
		releaseFirst();

		await expect(first).resolves.toMatchObject({ status: "failed" });
		await expect(second).resolves.toMatchObject({ status: "completed", text: "answer to B" });
		expect(calls).toBe(2);
		expect(prompts[1].at(-1)?.role).toBe("user");
		expect(JSON.stringify(prompts[1].at(-1)?.content)).toContain("B");
		const events = await storage.load("queue-fail-before-pickup");
		const queueId = (
			events.find(
				(event) => event.type === "user-message" && event.meta?.queued === true,
			) as Extract<StoredEvent, { type: "user-message" }>
		).meta?.queueId;
		expect(events.filter((event) => event.type === "run-start")).toHaveLength(2);
		expect(
			events.some(
				(event) =>
					event.type === "step" &&
					typeof queueId === "string" &&
					event.inputQueueIds?.includes(queueId),
			),
		).toBe(true);
		expect(replaySession(events).pendingMessages).toBe(0);
	});

	it("reconciles a queued final answer when the original run-end write fails", async () => {
		const modelId = "mock/queue-final-run-end-failure";
		setContextWindow(modelId, 100_000);
		const baseStorage = memoryStorage();
		let rejectedRunEnd = false;
		const storage = {
			async append(sessionId: string, event: StoredEvent) {
				if (event.type === "run-end" && !rejectedRunEnd) {
					rejectedRunEnd = true;
					throw new Error("run-end write failed");
				}
				await baseStorage.append(sessionId, event);
			},
			load: (sessionId: string) => baseStorage.load(sessionId),
		};
		let releaseFirst!: () => void;
		const firstGate = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});
		let markFirstStarted!: () => void;
		const firstStarted = new Promise<void>((resolve) => {
			markFirstStarted = resolve;
		});
		let markQueued!: () => void;
		const queued = new Promise<void>((resolve) => {
			markQueued = resolve;
		});
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				if (calls === 1) {
					markFirstStarted();
					await firstGate;
					return { stream: textStream("answer to A") };
				}
				return { stream: textStream("durable answer to B") };
			},
		});
		const agentic = createAgentic({
			storage,
			getModel: () => model,
			onEvent: (event) => {
				if (event.type === "queued-message") markQueued();
			},
		});
		const session = agentic.session("queue-final-run-end-failure", { model: modelId });

		const first = session.send("A");
		await firstStarted;
		const second = session.send("B");
		await queued;
		releaseFirst();

		await expect(first).rejects.toThrow("run-end write failed");
		await expect(second).resolves.toMatchObject({
			status: "completed",
			text: "durable answer to B",
		});
		expect(calls).toBe(2);
		const events = await storage.load("queue-final-run-end-failure");
		expect(events.filter((event) => event.type === "run-start")).toHaveLength(1);
		expect(events.filter((event) => event.type === "run-end")).toHaveLength(1);
		expect(replaySession(events).interruptedRunId).toBeNull();
		expect(replaySession(events).pendingMessages).toBe(0);
	});

	it("resumes the open run when terminal persistence fails before queued pickup", async () => {
		const modelId = "mock/queue-pending-run-end-failure";
		setContextWindow(modelId, 100_000);
		const baseStorage = memoryStorage();
		let rejectedRunEnd = false;
		const storage = {
			async append(sessionId: string, event: StoredEvent) {
				if (event.type === "run-end" && !rejectedRunEnd) {
					rejectedRunEnd = true;
					throw new Error("run-end write failed");
				}
				await baseStorage.append(sessionId, event);
			},
			load: (sessionId: string) => baseStorage.load(sessionId),
		};
		let releaseFirst!: () => void;
		const firstGate = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});
		let markFirstStarted!: () => void;
		const firstStarted = new Promise<void>((resolve) => {
			markFirstStarted = resolve;
		});
		let markQueued!: () => void;
		const queued = new Promise<void>((resolve) => {
			markQueued = resolve;
		});
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				if (calls === 1) {
					markFirstStarted();
					await firstGate;
					return {
						stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
							{ type: "stream-start", warnings: [] },
							{ type: "error", error: new Error("Invalid API key") },
						]),
					};
				}
				return { stream: textStream("recovered answer to B") };
			},
		});
		const agentic = createAgentic({
			storage,
			getModel: () => model,
			onEvent: (event) => {
				if (event.type === "queued-message") markQueued();
			},
		});
		const session = agentic.session("queue-pending-run-end-failure", { model: modelId });

		const first = session.send("A");
		await firstStarted;
		const second = session.send("B");
		await queued;
		releaseFirst();

		await expect(first).rejects.toThrow("run-end write failed");
		await expect(second).resolves.toMatchObject({
			status: "completed",
			text: "recovered answer to B",
		});
		expect(calls).toBe(2);
		const events = await storage.load("queue-pending-run-end-failure");
		expect(events.filter((event) => event.type === "run-start")).toHaveLength(1);
		expect(events.filter((event) => event.type === "run-resume")).toHaveLength(1);
		expect(events.filter((event) => event.type === "run-end")).toHaveLength(1);
		const replayed = replaySession(events);
		expect(replayed.interruptedRunId).toBeNull();
		expect(replayed.pendingMessages).toBe(0);
	});

	it("keeps a shared run alive when its initiator aborts after another caller queues", async () => {
		const modelId = "mock/queue-abort-after-inflight";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const abort = new AbortController();
		let releaseFirst!: () => void;
		const firstGate = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});
		let markFirstStarted!: () => void;
		const firstStarted = new Promise<void>((resolve) => {
			markFirstStarted = resolve;
		});
		let markQueued!: () => void;
		const queued = new Promise<void>((resolve) => {
			markQueued = resolve;
		});
		let calls = 0;
		let aborted = false;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				if (calls === 1) {
					markFirstStarted();
					await firstGate;
					return { stream: textStream("answer to A") };
				}
				return { stream: textStream("answer to B") };
			},
		});
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
			onEvent: (event) => {
				if (event.type === "queued-message") markQueued();
				if (event.type === "step" && !aborted) {
					aborted = true;
					abort.abort(new Error("stop A after its step"));
				}
			},
		});
		const session = agentic.session("queue-abort-after-inflight", { model: modelId });

		const first = session.send("A", { abortSignal: abort.signal });
		await firstStarted;
		const second = session.send("B");
		await queued;
		releaseFirst();

		await expect(first).resolves.toMatchObject({ status: "completed", text: "answer to B" });
		await expect(second).resolves.toMatchObject({ status: "completed", text: "answer to B" });
		const events = await storage.load("queue-abort-after-inflight");
		const steps = events.filter(
			(event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step",
		);
		expect(steps).toHaveLength(2);
		expect(steps[0].inputQueueIds).toEqual([]);
		expect(steps[1].inputQueueIds).toHaveLength(1);
		expect(calls).toBe(2);
		expect(replaySession(events).pendingMessages).toBe(0);
	});

	it("activates a queued listener only when a pass causally includes its input", async () => {
		const modelId = "mock/queue-listener-gating";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		let releaseFirst!: () => void;
		const firstGate = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});
		let markFirstStarted!: () => void;
		const firstStarted = new Promise<void>((resolve) => {
			markFirstStarted = resolve;
		});
		let markQueued!: () => void;
		const queued = new Promise<void>((resolve) => {
			markQueued = resolve;
		});
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				if (calls === 1) {
					markFirstStarted();
					await firstGate;
					return { stream: textStream("answer to A") };
				}
				return { stream: textStream("answer to B") };
			},
		});
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
			onEvent: (event) => {
				if (event.type === "queued-message") markQueued();
			},
		});
		const session = agentic.session("queue-listener-gating", { model: modelId });
		const queuedDeltas: string[] = [];

		const first = session.send("A");
		await firstStarted;
		const second = session.send("B", {
			onPart: (part) => {
				if (part.type === "text-delta") queuedDeltas.push(part.text);
			},
		});
		await queued;
		releaseFirst();

		await Promise.all([first, second]);
		expect(queuedDeltas).toEqual(["answer to B"]);
		expect(calls).toBe(2);
	});

	it("collapses multiple unseen queued sends into one recovery run and streams it to both", async () => {
		const modelId = "mock/queue-batch-recovery";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const recoveryWinnerAbort = new AbortController();
		let releaseFirst!: () => void;
		const firstGate = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});
		let markFirstStarted!: () => void;
		const firstStarted = new Promise<void>((resolve) => {
			markFirstStarted = resolve;
		});
		let queuedCount = 0;
		let markBothQueued!: () => void;
		const bothQueued = new Promise<void>((resolve) => {
			markBothQueued = resolve;
		});
		let releaseRecovery!: () => void;
		const recoveryGate = new Promise<void>((resolve) => {
			releaseRecovery = resolve;
		});
		let markRecoveryStarted!: () => void;
		const recoveryStarted = new Promise<void>((resolve) => {
			markRecoveryStarted = resolve;
		});
		let recoverySignal: AbortSignal | undefined;
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => {
				calls += 1;
				if (calls === 1) {
					markFirstStarted();
					await firstGate;
					return {
						stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
							{ type: "stream-start", warnings: [] },
							{ type: "error", error: new Error("Invalid API key") },
						]),
					};
				}
				recoverySignal = abortSignal;
				markRecoveryStarted();
				await recoveryGate;
				return { stream: textStream("answer to B and C") };
			},
		});
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
			onEvent: (event) => {
				if (event.type === "queued-message" && ++queuedCount === 2) markBothQueued();
			},
		});
		const session = agentic.session("queue-batch-recovery", { model: modelId });
		const bDeltas: string[] = [];
		const cDeltas: string[] = [];

		const first = session.send("A");
		await firstStarted;
		const second = session.send("B", {
			abortSignal: recoveryWinnerAbort.signal,
			onPart: (part) => {
				if (part.type === "text-delta") bDeltas.push(part.text);
			},
		});
		const third = session.send("C", {
			onPart: (part) => {
				if (part.type === "text-delta") cDeltas.push(part.text);
			},
		});
		await bothQueued;
		releaseFirst();
		await recoveryStarted;
		recoveryWinnerAbort.abort(new Error("B disconnected"));
		await Promise.resolve();
		expect(recoverySignal).toBeInstanceOf(AbortSignal);
		expect(recoverySignal?.aborted).toBe(false);
		releaseRecovery();

		await expect(first).resolves.toMatchObject({ status: "failed" });
		await expect(Promise.all([second, third])).resolves.toEqual([
			expect.objectContaining({ status: "completed", text: "answer to B and C" }),
			expect.objectContaining({ status: "completed", text: "answer to B and C" }),
		]);
		const events = await storage.load("queue-batch-recovery");
		const queuedIds = events.flatMap((event) =>
			event.type === "user-message" && typeof event.meta?.queueId === "string"
				? [event.meta.queueId]
				: [],
		);
		const recoverySteps = events.filter(
			(event) => event.type === "step" && event.inputQueueIds?.length === 2,
		);
		expect(calls).toBe(2);
		expect(events.filter((event) => event.type === "run-start")).toHaveLength(2);
		expect(recoverySteps).toHaveLength(1);
		expect((recoverySteps[0] as Extract<StoredEvent, { type: "step" }>).inputQueueIds).toEqual(
			queuedIds,
		);
		expect(bDeltas).toContain("answer to B and C");
		expect(cDeltas).toContain("answer to B and C");
		expect(replaySession(events).pendingMessages).toBe(0);
	});

	it("returns a recovered batch's persisted failure to every queued caller", async () => {
		const modelId = "mock/queue-batch-failure";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		let releaseFirst!: () => void;
		const firstGate = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});
		let markFirstStarted!: () => void;
		const firstStarted = new Promise<void>((resolve) => {
			markFirstStarted = resolve;
		});
		let queuedCount = 0;
		let markBothQueued!: () => void;
		const bothQueued = new Promise<void>((resolve) => {
			markBothQueued = resolve;
		});
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				if (calls === 1) {
					markFirstStarted();
					await firstGate;
				}
				if (calls === 2) {
					return {
						stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
							{ type: "stream-start", warnings: [] },
							{
								type: "tool-call",
								toolCallId: "touch-1",
								toolName: "touch",
								input: "{}",
							},
							{
								type: "finish",
								finishReason: { unified: "tool-calls", raw: "tool_calls" },
								usage: {
									inputTokens: {
										total: 10,
										noCache: 10,
										cacheRead: undefined,
										cacheWrite: undefined,
									},
									outputTokens: { total: 1, text: 0, reasoning: undefined },
								},
							},
						]),
					};
				}
				return {
					stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
						{ type: "stream-start", warnings: [] },
						{ type: "error", error: new Error("Invalid API key") },
					]),
				};
			},
		});
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
			onEvent: (event) => {
				if (event.type === "queued-message" && ++queuedCount === 2) markBothQueued();
			},
		});
		const session = agentic.session("queue-batch-failure", {
			model: modelId,
			tools: {
				touch: tool({ inputSchema: z.object({}), execute: async () => "done" }),
			},
		});

		const first = session.send("A");
		await firstStarted;
		const second = session.send("B");
		const third = session.send("C");
		await bothQueued;
		releaseFirst();

		await expect(first).resolves.toMatchObject({ status: "failed" });
		await expect(Promise.all([second, third])).resolves.toEqual([
			expect.objectContaining({
				status: "failed",
				error: expect.objectContaining({ message: "Invalid API key" }),
			}),
			expect.objectContaining({
				status: "failed",
				error: expect.objectContaining({ message: "Invalid API key" }),
			}),
		]);
		expect(calls).toBe(3);
		const events = await storage.load("queue-batch-failure");
		expect(
			events.filter(
				(event) =>
					event.type === "step" &&
					event.finishReason === "tool-calls" &&
					event.inputQueueIds?.length === 2,
			),
		).toHaveLength(1);
	});

	it("hoists the exact queued input ahead of a later queue:false send", async () => {
		const modelId = "mock/queue-exact-hoist";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		let releaseFirst!: () => void;
		const firstGate = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});
		let markFirstStarted!: () => void;
		const firstStarted = new Promise<void>((resolve) => {
			markFirstStarted = resolve;
		});
		let markQueued!: () => void;
		const queued = new Promise<void>((resolve) => {
			markQueued = resolve;
		});
		const prompts: ModelMessage[][] = [];
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async ({ prompt }) => {
				calls += 1;
				prompts.push(prompt as ModelMessage[]);
				if (calls === 1) {
					markFirstStarted();
					await firstGate;
					return {
						stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
							{ type: "stream-start", warnings: [] },
							{ type: "error", error: new Error("Invalid API key") },
						]),
					};
				}
				return { stream: textStream("answer to B then C") };
			},
		});
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
			onEvent: (event) => {
				if (event.type === "queued-message") markQueued();
			},
		});
		const session = agentic.session("queue-exact-hoist", { model: modelId });

		const first = session.send("A");
		await firstStarted;
		const queuedSend = session.send("B");
		await queued;
		const serializedSend = session.send("C", { queue: false });
		releaseFirst();

		await expect(first).resolves.toMatchObject({ status: "failed" });
		await expect(Promise.all([queuedSend, serializedSend])).resolves.toEqual([
			expect.objectContaining({ status: "completed", text: "answer to B then C" }),
			expect.objectContaining({ status: "completed", text: "answer to B then C" }),
		]);
		const userTail = prompts[1]
			.filter((message) => message.role === "user")
			.slice(-2)
			.map((message) => JSON.stringify(message.content));
		expect(userTail[0]).toContain("B");
		expect(userTail[1]).toContain("C");
		expect(calls).toBe(2);
		expect(replaySession(await storage.load("queue-exact-hoist")).pendingMessages).toBe(0);
	});

	it("restart replay recovers a queue ignored by a later in-flight step", async () => {
		const modelId = "mock/queue-restart";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		await storage.append("queue-restart", {
			type: "user-message",
			at: "t0",
			id: "u-a",
			message: { role: "user", content: "A" },
		});
		await storage.append("queue-restart", {
			type: "run-start",
			at: "t1",
			runId: "r1",
			model: modelId,
		});
		await storage.append("queue-restart", {
			type: "user-message",
			at: "t2",
			id: "q-b",
			message: { role: "user", content: "B" },
			meta: { queued: true, queueId: "q-b" },
		});
		await storage.append("queue-restart", {
			type: "step",
			at: "t3",
			runId: "r1",
			messages: stored({ role: "assistant", content: "answer to A" }),
			inputMessageIds: ["u-a"],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "stop",
			usage: usage(),
		});
		await storage.append("queue-restart", {
			type: "run-end",
			at: "t4",
			runId: "r1",
			status: "failed",
		});

		const prompts: ModelMessage[][] = [];
		const model = new MockLanguageModelV3({
			doStream: async ({ prompt }) => {
				prompts.push(prompt as ModelMessage[]);
				return { stream: textStream("answer to B") };
			},
		});
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
		});
		const session = agentic.session("queue-restart", { model: modelId });

		await expect(session.isInterrupted()).resolves.toBe(true);
		await expect(session.resume()).resolves.toMatchObject({
			status: "completed",
			text: "answer to B",
		});
		expect(prompts).toHaveLength(1);
		expect(JSON.stringify(prompts[0].at(-1)?.content)).toContain("B");
		const replayed = replaySession(await storage.load("queue-restart"));
		expect(replayed.pendingQueueIds).toEqual([]);
		expect(replayed.pendingMessages).toBe(0);
	});

	it("drops a late mailbox signal after its queued input was already answered", async () => {
		const modelId = "mock/queue-late-signal";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const mailbox = createMailbox();
		await storage.append("queue-late-signal", {
			type: "user-message",
			at: "t0",
			id: "u-a",
			message: { role: "user", content: "A" },
		});
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				if (calls === 1) {
					await storage.append("queue-late-signal", {
						type: "user-message",
						at: "t1",
						id: "q-b",
						message: { role: "user", content: "B" },
						meta: { queued: true, queueId: "q-b" },
					});
					await storage.append("queue-late-signal", {
						type: "user-message",
						at: "t2",
						id: "q-x",
						message: { role: "user", content: "X" },
						meta: { queued: true, queueId: "q-x" },
					});
					mailbox.tryEnqueue("q-x");
					return { stream: textStream("answer to A") };
				}
				// q-b's append is durable and this pass sees it before its delayed
				// in-process signal arrives.
				mailbox.tryEnqueue("q-b");
				return { stream: textStream("answer to B and X") };
			},
		});

		await expect(
			runLoop({
				sessionId: "queue-late-signal",
				agent: { model: modelId },
				storage,
				getModel: () => model,
				mailbox,
			}),
		).resolves.toMatchObject({ status: "completed", text: "answer to B and X" });
		expect(calls).toBe(2);
		expect(mailbox.queued).toEqual([]);
		const steps = (await storage.load("queue-late-signal")).filter(
			(event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step",
		);
		expect(steps[0].inputQueueIds).toEqual([]);
		expect(steps[1].inputQueueIds).toEqual(["q-b", "q-x"]);
	});

	it("runLoop folds a queued message in after a batched parallel tool-call step", async () => {
		setContextWindow("mock/model", 100_000);
		const storage = memoryStorage();
		const sessionId = "queue-tools";
		await storage.append(sessionId, {
			type: "user-message",
			at: "t0",
			id: "u-weather",
			message: { role: "user", content: "weather in san antonio and houston" },
		});

		const mailbox = createMailbox();
		// The queued send() lands while the batched tools are still running:
		// ledger append first (durable), then the mailbox signal.
		let queuedSent = false;
		const weather = tool({
			inputSchema: z.object({ city: z.string() }),
			execute: async ({ city }) => {
				if (!queuedSent) {
					queuedSent = true;
					await storage.append(sessionId, {
						type: "user-message",
						at: "t1",
						id: "q-1",
						message: { role: "user", content: "then email it to me" },
						meta: { queued: true, queueId: "q-1" },
					});
					mailbox.tryEnqueue("q-1");
				}
				return { city, temp: "78F" };
			},
		});

		const finish: LanguageModelV3StreamPart = {
			type: "finish",
			finishReason: { unified: "stop", raw: "stop" },
			usage: {
				inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
				outputTokens: { total: 5, text: 5, reasoning: undefined },
			},
		};
		const prompts: Array<Array<{ role: string; content: unknown }>> = [];
		const model = new MockLanguageModelV3({
			doStream: async ({ prompt }) => {
				prompts.push(prompt as Array<{ role: string; content: unknown }>);
				if (prompts.length === 1) {
					// one step, two parallel tool calls — a batch
					return {
						stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
							{ type: "stream-start", warnings: [] },
							{
								type: "tool-call",
								toolCallId: "sa",
								toolName: "weather",
								input: JSON.stringify({ city: "san antonio" }),
							},
							{
								type: "tool-call",
								toolCallId: "hou",
								toolName: "weather",
								input: JSON.stringify({ city: "houston" }),
							},
							{ ...finish, finishReason: { unified: "tool-calls", raw: "tool_calls" } },
						]),
					};
				}
				return {
					stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
						{ type: "stream-start", warnings: [] },
						{ type: "text-start", id: "1" },
						{ type: "text-delta", id: "1", delta: "78F in both, emailed" },
						{ type: "text-end", id: "1" },
						finish,
					]),
				};
			},
		});

		const result = await runLoop({
			sessionId,
			agent: { model: "mock/model", tools: { weather } },
			storage,
			getModel: () => model,
			mailbox,
		});

		expect(result.status).toBe("completed");
		expect(result.text).toBe("78F in both, emailed");
		// the queued arrival stopped the pass at the step boundary — after BOTH
		// batched tool results, never between them
		expect(prompts).toHaveLength(2);
		expect(prompts[1].map((m) => m.role)).toEqual(["user", "assistant", "tool", "user"]);
		// the batch survived replay intact: 2 calls paired with 2 results,
		// with the hoisted user message after the results, not inside the pair
		const parts = (m: { content: unknown }) => m.content as Array<{ type: string }>;
		expect(parts(prompts[1][1]).filter((p) => p.type === "tool-call")).toHaveLength(2);
		expect(parts(prompts[1][2]).filter((p) => p.type === "tool-result")).toHaveLength(2);
		expect(JSON.stringify(prompts[1][3].content)).toContain("then email it to me");
		// one run, two steps, closed cleanly, nothing left pending
		const events = await storage.load(sessionId);
		expect(events.filter((e) => e.type === "step")).toHaveLength(2);
		expect(events.filter((e) => e.type === "run-end")).toHaveLength(1);
		expect(mailbox.queued).toHaveLength(0);
		expect(replaySession(events).pendingMessages).toBe(0);
	});
});

describe("crash-window finalization", () => {
	async function seedFinalStep(
		storage: ReturnType<typeof memoryStorage>,
		sessionId: string,
		finishReason = "stop",
	): Promise<void> {
		await storage.append(sessionId, {
			type: "user-message",
			at: "t0",
			id: "u-a",
			message: { role: "user", content: "A" },
		});
		await storage.append(sessionId, {
			type: "run-start",
			at: "t1",
			runId: "r1",
			model: "mock/finalize",
		});
		await storage.append(sessionId, {
			type: "step",
			at: "t2",
			runId: "r1",
			messages: stored({ role: "assistant", content: "answer to A" }),
			inputMessageIds: ["u-a"],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason,
			usage: usage(),
		});
	}

	it("manual resume closes a persisted final answer without duplicating it", async () => {
		setContextWindow("mock/finalize", 100_000);
		const storage = memoryStorage();
		await seedFinalStep(storage, "manual-finalize");
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				throw new Error("model should not run");
			},
		});
		const session = createAgentic({ storage, getModel: () => model }).session("manual-finalize", {
			model: "mock/finalize",
		});

		await expect(session.resume()).resolves.toMatchObject({
			status: "completed",
			text: "answer to A",
		});
		expect(calls).toBe(0);
		const events = await storage.load("manual-finalize");
		expect(events.at(-1)).toMatchObject({ type: "run-end", runId: "r1", status: "completed" });
		expect(replaySession(events).interruptedRunId).toBeNull();
	});

	it("does not return stale text from an earlier run when finalizing a textless answer", async () => {
		setContextWindow("mock/finalize", 100_000);
		const storage = memoryStorage();
		await storage.append("textless-finalize", {
			type: "user-message",
			at: "t0",
			id: "u-old",
			message: { role: "user", content: "old question" },
		});
		await storage.append("textless-finalize", {
			type: "run-start",
			at: "t1",
			runId: "old-run",
			model: "mock/finalize",
		});
		await storage.append("textless-finalize", {
			type: "step",
			at: "t2",
			runId: "old-run",
			messages: stored({ role: "assistant", content: "stale old answer" }),
			inputMessageIds: ["u-old"],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "stop",
			usage: usage(),
		});
		await storage.append("textless-finalize", {
			type: "run-end",
			at: "t3",
			runId: "old-run",
			status: "completed",
		});
		await storage.append("textless-finalize", {
			type: "user-message",
			at: "t4",
			id: "u-current",
			message: { role: "user", content: "current question" },
		});
		await storage.append("textless-finalize", {
			type: "run-start",
			at: "t5",
			runId: "current-run",
			model: "mock/finalize",
		});
		await storage.append("textless-finalize", {
			type: "step",
			at: "t6",
			runId: "current-run",
			messages: stored({ role: "assistant", content: "" }),
			inputMessageIds: ["u-current"],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "stop",
			usage: usage(),
		});
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				throw new Error("model should not run");
			},
		});
		const session = createAgentic({ storage, getModel: () => model }).session("textless-finalize", {
			model: "mock/finalize",
		});

		await expect(session.resume()).resolves.toMatchObject({ status: "completed", text: "" });
		expect(calls).toBe(0);
	});

	it("auto-resume finalizes a persisted clean step without a model call", async () => {
		setContextWindow("mock/finalize", 100_000);
		const storage = memoryStorage();
		await seedFinalStep(storage, "auto-finalize", "content-filter");
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				throw new Error("model should not run");
			},
		});
		const agentic = createAgentic({
			storage,
			getModel: () => model,
			autoResume: () => ({ model: "mock/finalize" }),
		});

		await agentic.resumeInterrupted();
		expect(calls).toBe(0);
		expect(replaySession(await storage.load("auto-finalize")).interruptedRunId).toBeNull();
	});

	it("a new send closes the old final run before starting exactly one new model turn", async () => {
		setContextWindow("mock/finalize", 100_000);
		const storage = memoryStorage();
		await seedFinalStep(storage, "send-after-finalize");
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				return {
					stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
						{ type: "stream-start", warnings: [] },
						{ type: "text-start", id: "1" },
						{ type: "text-delta", id: "1", delta: "answer to B" },
						{ type: "text-end", id: "1" },
						{
							type: "finish",
							finishReason: { unified: "stop", raw: "stop" },
							usage: {
								inputTokens: {
									total: 1,
									noCache: 1,
									cacheRead: undefined,
									cacheWrite: undefined,
								},
								outputTokens: { total: 1, text: 1, reasoning: undefined },
							},
						},
					]),
				};
			},
		});
		const session = createAgentic({ storage, getModel: () => model }).session(
			"send-after-finalize",
			{ model: "mock/finalize" },
		);

		await expect(session.send("B")).resolves.toMatchObject({
			status: "completed",
			text: "answer to B",
		});
		expect(calls).toBe(1);
		const events = await storage.load("send-after-finalize");
		expect(events.filter((event) => event.type === "run-start")).toHaveLength(2);
		expect(events.filter((event) => event.type === "run-end")).toHaveLength(2);
		expect(replaySession(events).messages.map((entry) => entry.message.role)).toEqual([
			"user",
			"assistant",
			"user",
			"assistant",
		]);
	});
});

describe("hoistSandwichedUsers", () => {
	const user = (content: string): ModelMessage => ({ role: "user", content });
	const toolCalls = (...ids: string[]): ModelMessage => ({
		role: "assistant",
		content: ids.map((id) => ({
			type: "tool-call",
			toolCallId: id,
			toolName: "lookup",
			input: {},
		})),
	});
	const toolResults = (...ids: string[]): ModelMessage => ({
		role: "tool",
		content: ids.map((id) => ({
			type: "tool-result",
			toolCallId: id,
			toolName: "lookup",
			output: { type: "text", value: "ok" },
		})),
	});
	const roles = (messages: ModelMessage[]) => messages.map((m) => m.role);

	it("moves a sandwiched user past a tool-call/result pair without splitting it", () => {
		const messages = [user("question"), user("queued"), toolCalls("a"), toolResults("a")];
		hoistSandwichedUsers(messages, 1);
		expect(roles(messages)).toEqual(["user", "assistant", "tool", "user"]);
		expect(messages[3]).toEqual(user("queued"));
	});

	it("treats a batched parallel tool-call step as one block", () => {
		const messages = [user("question"), user("queued"), toolCalls("a", "b"), toolResults("a", "b")];
		hoistSandwichedUsers(messages, 1);
		expect(roles(messages)).toEqual(["user", "assistant", "tool", "user"]);
		expect(messages[3]).toEqual(user("queued"));
	});

	it("hoists past multiple trailing steps", () => {
		// the message raced past a load snapshot, so two whole steps landed on top
		const messages = [
			user("question"),
			user("queued"),
			toolCalls("a"),
			toolResults("a"),
			toolCalls("b"),
			toolResults("b"),
		];
		hoistSandwichedUsers(messages, 1);
		expect(roles(messages)).toEqual(["user", "assistant", "tool", "assistant", "tool", "user"]);
		expect(messages[5]).toEqual(user("queued"));
	});

	it("lifts only the queued tail of the user block, preserving order", () => {
		const messages = [user("answered"), user("q1"), user("q2"), toolCalls("a"), toolResults("a")];
		hoistSandwichedUsers(messages, 2);
		expect(roles(messages)).toEqual(["user", "assistant", "tool", "user", "user"]);
		expect(messages[0]).toEqual(user("answered"));
		expect(messages[3]).toEqual(user("q1"));
		expect(messages[4]).toEqual(user("q2"));
	});

	it("is a no-op when input is already trailing or max is 0", () => {
		const trailing = [user("question"), toolCalls("a"), toolResults("a"), user("queued")];
		hoistSandwichedUsers(trailing, 1);
		expect(roles(trailing)).toEqual(["user", "assistant", "tool", "user"]);

		const untouched = [user("question"), user("queued"), toolCalls("a"), toolResults("a")];
		hoistSandwichedUsers(untouched, 0);
		expect(roles(untouched)).toEqual(["user", "user", "assistant", "tool"]);
	});

	it("is a no-op when there is no user block to hoist", () => {
		const messages = [toolCalls("a"), toolResults("a")];
		hoistSandwichedUsers(messages, 1);
		expect(roles(messages)).toEqual(["assistant", "tool"]);
		hoistSandwichedUsers([], 1);
	});
});

describe("auto-resume", () => {
	async function waitUntil(predicate: () => boolean | Promise<boolean>): Promise<void> {
		for (let attempt = 0; attempt < 100; attempt++) {
			if (await predicate()) return;
			await new Promise((resolve) => setTimeout(resolve, 0));
		}
		throw new Error("condition was not reached");
	}

	const textStream = (text: string) =>
		convertArrayToReadableStream<LanguageModelV3StreamPart>([
			{ type: "stream-start", warnings: [] },
			{ type: "text-start", id: "1" },
			{ type: "text-delta", id: "1", delta: text },
			{ type: "text-end", id: "1" },
			{
				type: "finish",
				finishReason: { unified: "stop", raw: "stop" },
				usage: {
					inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
					outputTokens: { total: 5, text: 5, reasoning: undefined },
				},
			},
		]);

	it("boot sweep finds a crashed run and drives it to completion", async () => {
		setContextWindow("mock/model", 100_000);
		const storage = memoryStorage();
		await storage.append("crashed", {
			type: "user-message",
			at: "t0",
			id: "u-1",
			message: { role: "user", content: "count to three" },
		});
		await storage.append("crashed", {
			type: "run-start",
			at: "t1",
			runId: "r1",
			model: "mock/model",
		});
		await storage.append("crashed", {
			type: "step",
			at: "t2",
			runId: "r1",
			messages: stored({ role: "assistant", content: "one…" }),
			inputMessageIds: ["u-1"],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "tool-calls",
			usage: usage(),
		});
		// the process died here — no run-end

		const model = new MockLanguageModelV3({
			doStream: async () => ({ stream: textStream("two, three — done") }),
		});
		const eventTypes: string[] = [];
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
			autoResume: () => ({ model: "mock/model" }),
			onEvent: (e) => eventTypes.push(e.type),
		});

		// The background boot sweep and this call race benignly: kicks re-check
		// under the session lock, so exactly one of them re-drives the run.
		await agentic.resumeInterrupted();

		const ledger = await storage.load("crashed");
		expect(ledger.filter((e) => e.type === "run-resume")).toEqual([
			expect.objectContaining({ type: "run-resume", runId: "r1", auto: true }),
		]);
		// resumed under the ORIGINAL runId — no second run-start
		expect(ledger.filter((e) => e.type === "run-start")).toHaveLength(1);
		expect(ledger.filter((e) => e.type === "run-end")).toEqual([
			expect.objectContaining({ status: "completed" }),
		]);
		expect(eventTypes).toContain("auto-resume");
		expect(await agentic.interruptedSessions()).toEqual([]);
	});

	it("public cancellation stops an in-process auto-resumed run", async () => {
		const modelId = "mock/cancel-auto-resumed-live";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		await storage.append("cancel-auto-resumed-live", {
			type: "user-message",
			at: "t0",
			id: "u-1",
			message: { role: "user", content: "continue after restart" },
		});
		await storage.append("cancel-auto-resumed-live", {
			type: "run-start",
			at: "t1",
			runId: "crashed-run",
			model: modelId,
		});
		let markStarted!: () => void;
		const started = new Promise<void>((resolve) => {
			markStarted = resolve;
		});
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => ({
				stream: new ReadableStream<LanguageModelV3StreamPart>({
					start(controller) {
						const stop = () => controller.error(abortSignal?.reason ?? new Error("aborted"));
						abortSignal?.addEventListener("abort", stop, { once: true });
						controller.enqueue({ type: "stream-start", warnings: [] });
						controller.enqueue({ type: "text-start", id: "partial" });
						controller.enqueue({
							type: "text-delta",
							id: "partial",
							delta: "auto-resumed partial",
						});
						markStarted();
					},
				}),
			}),
		});
		const agentic = createAgentic({
			storage,
			getModel: () => model,
			autoResume: (sessionId) =>
				sessionId === "cancel-auto-resumed-live" ? { model: modelId } : null,
		});

		await started;
		expect(agentic.isRunning("cancel-auto-resumed-live")).toBe(true);
		expect(agentic.cancel("cancel-auto-resumed-live", new Error("stop recovered generation"))).toBe(
			true,
		);
		await waitUntil(() => !agentic.isRunning("cancel-auto-resumed-live"));

		const events = await storage.load("cancel-auto-resumed-live");
		expect(events.at(-1)).toMatchObject({
			type: "run-end",
			runId: "crashed-run",
			status: "cancelled",
			error: { message: "stop recovered generation" },
		});
	});

	it("manual sweep resumes an orphaned queued message and returns the result", async () => {
		setContextWindow("mock/model", 100_000);
		const storage = memoryStorage();
		const model = new MockLanguageModelV3({ doStream: async () => ({ stream: textStream("42") }) });
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
			autoResume: (sessionId) => (sessionId === "orphan" ? { model: "mock/model" } : null),
		});
		// let the (empty) boot sweep drain before creating the orphan
		await new Promise((resolve) => setTimeout(resolve, 0));

		await storage.append("orphan", {
			type: "user-message",
			at: "t0",
			id: "q-1",
			message: { role: "user", content: "what is 6*7?" },
			meta: { queued: true, queueId: "q-1" },
		});

		const results = await agentic.resumeInterrupted();
		expect(results).toHaveLength(1);
		expect(results[0].status).toBe("completed");
		expect(results[0].text).toBe("42");
		expect(replaySession(await storage.load("orphan")).pendingMessages).toBe(0);
	});

	it("gives up after the ledger-counted attempt cap instead of crash-looping", async () => {
		setContextWindow("mock/model", 100_000);
		const storage = memoryStorage();
		await storage.append("wedged", {
			type: "run-start",
			at: "t0",
			runId: "r1",
			model: "mock/model",
		});
		for (let i = 0; i < 3; i++) {
			await storage.append("wedged", { type: "run-resume", at: "t", runId: "r1", auto: true });
		}

		let modelCalls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				modelCalls += 1;
				return { stream: textStream("never") };
			},
		});
		const giveUps: string[] = [];
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
			autoResume: () => ({ model: "mock/model" }),
			onEvent: (e) => {
				if (e.type === "auto-resume" && e.action === "give-up") giveUps.push(e.sessionId);
			},
		});

		const results = await agentic.resumeInterrupted();
		expect(results).toEqual([]);
		expect(giveUps).toContain("wedged");
		expect(modelCalls).toBe(0);
		const events = await storage.load("wedged");
		expect(events.filter((e) => e.type === "run-resume")).toHaveLength(3);
		expect(events.at(-1)).toMatchObject({
			type: "run-end",
			runId: "r1",
			status: "failed",
		});
		expect(replaySession(events).interruptedRunId).toBeNull();
		expect(await agentic.interruptedSessions()).toEqual([]);
	});

	it("does not discard queued-only work when automatic attempts are exhausted", async () => {
		const storage = memoryStorage();
		await storage.append("queued-give-up", {
			type: "user-message",
			at: "t0",
			id: "q-1",
			message: { role: "user", content: "keep me" },
			meta: { queued: true, queueId: "q-1" },
		});
		for (let attempt = 0; attempt < 3; attempt++) {
			await storage.append("queued-give-up", {
				type: "run-resume",
				at: "t",
				runId: null,
				auto: true,
			});
			await storage.append("queued-give-up", {
				type: "run-end",
				at: "t",
				runId: `failed-${attempt}`,
				status: "failed",
			});
		}
		const agentic = createAgentic({
			storage,
			autoResume: () => ({ model: "mock/model" }),
			getModel: () => {
				throw new Error("model should not run");
			},
		});

		await expect(agentic.resumeInterrupted()).resolves.toEqual([]);
		const replayed = replaySession(await storage.load("queued-give-up"));
		expect(replayed.pendingQueueIds).toEqual(["q-1"]);
		expect(replayed.pendingMessages).toBe(1);
		expect(await agentic.interruptedSessions()).toEqual(["queued-give-up"]);
	});

	it.each([
		["configured", 2, 2],
		["default", undefined, 4],
	] as const)("honors the %s auto-resume concurrency cap", async (_label, configured, cap) => {
		const modelId = `mock/concurrency-${configured ?? "default"}`;
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		let active = 0;
		let peak = 0;
		const releases: Array<() => void> = [];
		const model = new MockLanguageModelV3({
			doStream: async () => {
				active += 1;
				peak = Math.max(peak, active);
				await new Promise<void>((resolve) => releases.push(resolve));
				active -= 1;
				return { stream: textStream("resumed") };
			},
		});
		const agentFor = () => ({ model: modelId });
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
			autoResume: configured === undefined ? { agentFor } : { agentFor, maxConcurrent: configured },
		});
		// Let the automatic sweep observe the initially empty storage.
		await new Promise((resolve) => setTimeout(resolve, 0));

		const sessionCount = cap + 2;
		for (let index = 0; index < sessionCount; index++) {
			await storage.append(`concurrent-${configured ?? "default"}-${index}`, {
				type: "user-message",
				at: "t0",
				id: "u-resume",
				message: { role: "user", content: "resume me" },
			});
		}

		const sweep = agentic.resumeInterrupted();
		await waitUntil(() => releases.length === cap);
		await new Promise((resolve) => setTimeout(resolve, 0));
		expect(active).toBe(cap);
		expect(releases).toHaveLength(cap);

		let released = 0;
		while (released < sessionCount) {
			await waitUntil(() => releases.length > released);
			const ready = releases.length;
			for (const release of releases.slice(released, ready)) release();
			released = ready;
		}
		await expect(sweep).resolves.toHaveLength(sessionCount);
		expect(peak).toBe(cap);
	});

	it("shares the concurrency cap across racing boot and manual sweeps", async () => {
		const modelId = "mock/concurrency-racing-sweeps";
		setContextWindow(modelId, 100_000);
		const storage = memoryStorage();
		const sessionCount = 4;
		for (let index = 0; index < sessionCount; index++) {
			await storage.append(`racing-sweep-${index}`, {
				type: "user-message",
				at: "t0",
				id: "u-resume",
				message: { role: "user", content: "resume me" },
			});
		}

		let active = 0;
		let peak = 0;
		const releases: Array<() => void> = [];
		const model = new MockLanguageModelV3({
			doStream: async () => {
				active += 1;
				peak = Math.max(peak, active);
				await new Promise<void>((resolve) => releases.push(resolve));
				active -= 1;
				return { stream: textStream("resumed") };
			},
		});
		const agentic = createAgentic({
			apiKey: "test",
			storage,
			getModel: () => model,
			autoResume: { agentFor: () => ({ model: modelId }), maxConcurrent: 2 },
		});
		await waitUntil(() => releases.length === 2);
		// While the boot sweep holds both slots, a manual sweep sees the other
		// sessions but must share the same runtime-wide cap.
		const manualSweep = agentic.resumeInterrupted();
		await new Promise((resolve) => setTimeout(resolve, 0));
		expect(releases).toHaveLength(2);

		let released = 0;
		while (released < sessionCount) {
			await waitUntil(() => releases.length > released);
			const ready = releases.length;
			for (const release of releases.slice(released, ready)) release();
			released = ready;
		}
		await manualSweep;
		await waitUntil(async () => (await agentic.interruptedSessions()).length === 0);
		expect(peak).toBe(2);
	});

	it("rejects a non-positive auto-resume concurrency cap", async () => {
		const agentic = createAgentic({
			storage: memoryStorage(),
			autoResume: { agentFor: () => null, maxConcurrent: 0 },
		});
		await expect(agentic.resumeInterrupted()).rejects.toThrow(
			"autoResume.maxConcurrent must be a positive integer",
		);
	});
});

describe("message identity", () => {
	const finishPart: LanguageModelV3StreamPart = {
		type: "finish",
		finishReason: { unified: "stop", raw: "stop" },
		usage: {
			inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
			outputTokens: { total: 5, text: 5, reasoning: undefined },
		},
	};
	const textStream = (text: string) =>
		convertArrayToReadableStream<LanguageModelV3StreamPart>([
			{ type: "stream-start", warnings: [] },
			{ type: "text-start", id: "1" },
			{ type: "text-delta", id: "1", delta: text },
			{ type: "text-end", id: "1" },
			finishPart,
		]);

	it("mints ids and timestamps for every message, stable across replays", async () => {
		setContextWindow("mock/identity", 100_000);
		const storage = memoryStorage();
		const model = new MockLanguageModelV3({
			doStream: async () => ({ stream: textStream("hi there") }),
		});
		const events: AgenticEvent[] = [];
		const agentic = createAgentic({
			storage,
			getModel: () => model,
			onEvent: (event) => events.push(event),
		});
		const session = agentic.session("identity", { model: "mock/identity" });
		const result = await session.send("hello");

		const messages = await session.messages();
		expect(messages.map((entry) => entry.message.role)).toEqual(["user", "assistant"]);
		for (const entry of messages) {
			expect(typeof entry.id).toBe("string");
			expect(entry.at.length).toBeGreaterThan(0);
		}
		expect(new Set(messages.map((entry) => entry.id)).size).toBe(2);

		// The ledger carries the same identities, on the event for user input and
		// per stored message for model output.
		const ledger = await storage.load("identity");
		const userEvent = ledger.find(
			(event): event is Extract<StoredEvent, { type: "user-message" }> =>
				event.type === "user-message",
		);
		expect(userEvent?.id).toBe(messages[0].id);
		expect(messages[0].at).toBe(userEvent?.at);
		const stepEvent = ledger.find(
			(event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step",
		);
		expect(stepEvent?.messages.map((entry) => entry.id)).toEqual([messages[1].id]);
		expect(messages[1].at).toBe(stepEvent?.at);

		// The RunResult names the run that produced it, closing the correlation
		// loop: result → run events → step messages.
		expect(typeof result.runId).toBe("string");
		expect(result.runId).toBe(stepEvent?.runId);

		// Replay is deterministic: same ids every load.
		expect((await session.messages()).map((entry) => entry.id)).toEqual(
			messages.map((entry) => entry.id),
		);

		// Live events are stamped, and the step event carries the persisted
		// identified messages.
		expect(events.length).toBeGreaterThan(0);
		expect(events.every((event) => typeof event.at === "string" && event.at.length > 0)).toBe(true);
		const liveStep = events.find(
			(event): event is Extract<AgenticEvent, { type: "step" }> => event.type === "step",
		);
		expect(liveStep?.messages.map((entry) => entry.id)).toEqual([messages[1].id]);
	});

	it("keeps retained ids and times across compaction; the summary gets a fresh id", async () => {
		const modelId = "mock/identity-compaction";
		setContextWindow(modelId, 100);
		const storage = memoryStorage();
		const sessionId = "identity-compaction";
		await storage.append(sessionId, {
			type: "user-message",
			at: "t0",
			id: "u-old",
			message: { role: "user", content: "old question" },
		});
		await storage.append(sessionId, {
			type: "step",
			at: "t1",
			runId: "old",
			messages: [{ id: "a-old", message: { role: "assistant", content: "old answer" } }],
			inputMessageIds: ["u-old"],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "stop",
			usage: usage({ inputTokens: 60, outputTokens: 20, totalTokens: 80 }),
		});
		await storage.append(sessionId, {
			type: "run-end",
			at: "t2",
			runId: "old",
			status: "completed",
		});
		await storage.append(sessionId, {
			type: "user-message",
			at: "t3",
			id: "u-current",
			message: { role: "user", content: "current question" },
		});
		const model = new MockLanguageModelV3({
			doGenerate: async () => ({
				content: [{ type: "text" as const, text: "summary of old q&a" }],
				finishReason: { unified: "stop" as const, raw: "stop" },
				usage: {
					inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
					outputTokens: { total: 5, text: 5, reasoning: undefined },
				},
				warnings: [],
			}),
			doStream: async () => ({ stream: textStream("current answer") }),
		});

		await expect(
			runLoop({
				sessionId,
				agent: { model: modelId, compaction: { limit: 0.5, keepRecent: 0 } },
				storage,
				getModel: () => model,
			}),
		).resolves.toMatchObject({ status: "completed", text: "current answer" });

		const events = await storage.load(sessionId);
		const compaction = events.find(
			(event): event is Extract<StoredEvent, { type: "compaction" }> => event.type === "compaction",
		);
		// The pending input survives the rebase with its exact identity — id AND
		// original persist time — while the fresh summary message gets a new id.
		const summaryEntry = compaction?.messages[0];
		const retainedEntry = compaction?.messages.find((entry) => entry.id === "u-current");
		expect(retainedEntry?.at).toBe("t3");
		expect(typeof summaryEntry?.id).toBe("string");
		expect(["u-old", "a-old", "u-current"]).not.toContain(summaryEntry?.id);

		const replayed = replaySession(events);
		expect(replayed.pendingMessages).toBe(0);
		const current = replayed.messages.find((entry) => entry.id === "u-current");
		expect(current?.at).toBe("t3");
		expect(JSON.stringify(current?.message.content)).toContain("current question");
	});
});
