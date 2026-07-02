import { describe, expect, it } from "vitest";
import { retryDelayMs } from "./backoff.js";
import { classifyFailure } from "./failure.js";
import { replaySession } from "./replay.js";
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
