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
import type { StepUsage, StoredEvent } from "./types.js";
import { reconcileBilledCost } from "./usage.js";

const usage = (partial: Partial<StepUsage> = {}): StepUsage => ({
	inputTokens: 100,
	outputTokens: 20,
	totalTokens: 120,
	cachedInputTokens: null,
	reasoningTokens: null,
	cost: 0.001,
	upstreamCost: null,
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

describe("reconcileBilledCost (BYOK)", () => {
	it("sums credits cost and upstream cost", () => {
		expect(reconcileBilledCost(0.001, null)).toBe(0.001); // OpenRouter credits
		expect(reconcileBilledCost(0, 0.0005)).toBe(0.0005); // BYOK, no fee
		expect(reconcileBilledCost(0.000025, 0.0005)).toBe(0.000525); // BYOK + 5% fee
		expect(reconcileBilledCost(null, null)).toBeNull();
	});
});

describe("replaySession", () => {
	it("rebuilds messages, detects interrupted runs, resets context on compaction", () => {
		const events: StoredEvent[] = [
			{ type: "user-message", at: "t", message: { role: "user", content: "hi" } },
			{ type: "run-start", at: "t", runId: "r1", model: "m" },
			{
				type: "step",
				at: "t",
				runId: "r1",
				messages: [{ role: "assistant", content: "hello" }],
				finishReason: "stop",
				usage: usage(),
			},
			{ type: "run-end", at: "t", runId: "r1", status: "completed" },
			{
				type: "compaction",
				at: "t",
				messages: [{ role: "user", content: "<summary>" }],
				usage: usage({ inputTokens: 5000, cost: 0.002, billedCost: 0.002 }),
			},
			{ type: "user-message", at: "t", message: { role: "user", content: "again" } },
			{ type: "run-start", at: "t", runId: "r2", model: "m" },
		];
		const replayed = replaySession(events);
		expect(replayed.messages.map((m) => m.content)).toEqual(["<summary>", "again"]);
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
			messages: [{ role: "assistant", content: "hello" }],
			finishReason: "stop",
			usage: usage(),
		};
		const user = (content: string): StoredEvent => ({
			type: "user-message",
			at: "t",
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

	it("resumes an interrupted task under the original run id", async () => {
		setContextWindow("mock/task-resume", 100_000);
		const storage = memoryStorage();
		await storage.append("interrupted-task", {
			type: "user-message",
			at: "t0",
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
});

describe("message queueing", () => {
	it("mailbox refuses enqueues once the run commits to ending", () => {
		const mailbox = createMailbox();
		expect(mailbox.tryEnqueue("q-1")).toBe(true);
		expect(mailbox.queued).toEqual(["q-1"]);
		mailbox.accepting = false;
		expect(mailbox.tryEnqueue("q-2")).toBe(false);
		expect(mailbox.queued).toEqual(["q-1"]);
	});

	it("runLoop answers a message queued mid-run, hoisted after model output", async () => {
		setContextWindow("mock/model", 100_000);
		const storage = memoryStorage();
		const sessionId = "queue-live";
		await storage.append(sessionId, {
			type: "user-message",
			at: "t0",
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
		expect(replaySession(events).pendingMessages).toBe(0);
	});

	it("runLoop folds a queued message in after a batched parallel tool-call step", async () => {
		setContextWindow("mock/model", 100_000);
		const storage = memoryStorage();
		const sessionId = "queue-tools";
		await storage.append(sessionId, {
			type: "user-message",
			at: "t0",
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
			messages: [{ role: "assistant", content: "one…" }],
			finishReason: "stop",
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
		// the ledger is untouched — a manual resume() can still retry by hand
		expect((await storage.load("wedged")).filter((e) => e.type === "run-resume")).toHaveLength(3);
	});
});
