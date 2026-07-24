import { describe, expect, it } from "vitest";
import { projectRun } from "./projection.js";
import type { StepUsage, StoredEvent } from "./types.js";

const usage: StepUsage = {
	inputTokens: 1,
	outputTokens: 1,
	totalTokens: 2,
	cachedInputTokens: null,
	reasoningTokens: null,
	cost: null,
	upstreamCost: null,
	isByok: null,
	billedCost: null,
};

function step(
	at: string,
	inputMessageIds: string[],
	messageId: string,
	content: string,
): Extract<StoredEvent, { type: "step" }> {
	return {
		type: "step",
		at,
		runId: "run",
		inputMessageIds,
		inputQueueIds: [],
		acknowledgesInput: true,
		finishReason: "stop",
		usage,
		messages: [{ id: messageId, message: { role: "assistant", content } }],
	};
}

describe("projectRun", () => {
	it("turns queued input into a new causal response segment", () => {
		const events: StoredEvent[] = [
			{ type: "user-message", at: "t0", id: "a", message: { role: "user", content: "A" } },
			{ type: "run-start", at: "t1", runId: "run", model: "model" },
			step("t2", ["a"], "answer-a", "Answer A"),
			{
				type: "user-message",
				at: "t3",
				id: "b",
				message: { role: "user", content: "B" },
				meta: { queued: true, queueId: "b" },
			},
			step("t4", ["b"], "answer-b", "Answer B"),
			{ type: "run-end", at: "t5", runId: "run", status: "completed" },
		];

		const projected = projectRun(events, "run");
		expect(projected).toMatchObject({ status: "completed", pendingInputs: [] });
		expect(projected?.segments.map((segment) => segment.inputs.map((input) => input.id))).toEqual([
			["a"],
			["b"],
		]);
		expect(
			projected?.segments.map((segment) =>
				segment.messages.map((message) => message.message.content),
			),
		).toEqual([["Answer A"], ["Answer B"]]);
	});

	it("keeps a queued input pending when an error step did not acknowledge it", () => {
		const events: StoredEvent[] = [
			{ type: "user-message", at: "t0", id: "a", message: { role: "user", content: "A" } },
			{ type: "run-start", at: "t1", runId: "run", model: "model" },
			{
				type: "user-message",
				at: "t2",
				id: "b",
				message: { role: "user", content: "B" },
				meta: { queued: true, queueId: "b" },
			},
			{
				...step("t3", ["a", "b"], "partial", "partial"),
				acknowledgesInput: false,
				finishReason: "error",
			},
			{
				type: "run-end",
				at: "t4",
				runId: "run",
				status: "failed",
				error: { message: "provider failed" },
			},
		];

		const projected = projectRun(events, "run");
		expect(projected?.segments[0]?.inputs.map((input) => input.id)).toEqual(["a"]);
		expect(projected?.pendingInputs.map((input) => input.id)).toEqual(["b"]);
		expect(projected?.segments[0]?.status).toBe("failed");
	});

	it("projects a pending input consumed by a successor run", () => {
		const events: StoredEvent[] = [
			{ type: "user-message", at: "t0", id: "a", message: { role: "user", content: "A" } },
			{ type: "run-start", at: "t1", runId: "first", model: "model" },
			{
				type: "user-message",
				at: "t2",
				id: "b",
				message: { role: "user", content: "B" },
				meta: { queued: true, queueId: "b" },
			},
			{
				type: "run-end",
				at: "t3",
				runId: "first",
				status: "failed",
				error: { message: "first run failed" },
			},
			{ type: "run-start", at: "t4", runId: "run", model: "model" },
			step("t5", ["b"], "answer-b", "Answer B"),
			{ type: "run-end", at: "t6", runId: "run", status: "completed" },
		];

		const projected = projectRun(events, "run");
		expect(projected?.segments.map((segment) => segment.inputs.map((input) => input.id))).toEqual([
			["b"],
		]);
		expect(projected?.pendingInputs).toEqual([]);
	});
});
