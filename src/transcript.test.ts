import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { LanguageModelV3StreamPart } from "@ai-sdk/provider";
import { convertArrayToReadableStream, MockLanguageModelV3 } from "ai/test";
import { afterEach, describe, expect, it } from "vitest";
import { createAgentic } from "./agentic.js";
import { setContextWindow } from "./modelMeta.js";
import {
	projectSession,
	type SessionTranscript,
	type TranscriptItem,
	type TranscriptResponseItem,
	type TranscriptUserItem,
	textOf,
} from "./projection.js";
import { createMailbox, runLoop } from "./run.js";
import { fileStorage, memoryStorage } from "./storage.js";
import type { StepUsage, StoredEvent } from "./types.js";

const tempDirs: string[] = [];

afterEach(async () => {
	await Promise.all(tempDirs.splice(0).map((dir) => fsp.rm(dir, { recursive: true, force: true })));
});

// ── ledger + stream fixtures ────────────────────────────────────────────

const usage = (partial: Partial<StepUsage> = {}): StepUsage => ({
	inputTokens: 100,
	outputTokens: 20,
	totalTokens: 120,
	cachedInputTokens: null,
	reasoningTokens: null,
	cost: null,
	upstreamCost: null,
	isByok: null,
	billedCost: null,
	...partial,
});

const user = (
	at: string,
	id: string,
	content: string,
	meta?: Record<string, unknown>,
): StoredEvent => ({
	type: "user-message",
	at,
	id,
	message: { role: "user", content },
	...(meta ? { meta } : {}),
});

const ackStep = (
	at: string,
	runId: string,
	inputMessageIds: string[],
	messageId: string,
	content: string,
): StoredEvent => ({
	type: "step",
	at,
	runId,
	inputMessageIds,
	inputQueueIds: [],
	acknowledgesInput: true,
	finishReason: "stop",
	usage: usage(),
	messages: [{ id: messageId, message: { role: "assistant", content } }],
});

const finishPart = (
	reason: "stop" | "tool-calls" = "stop",
	tokens = { input: 10, output: 5 },
): LanguageModelV3StreamPart => ({
	type: "finish",
	finishReason: { unified: reason, raw: reason },
	usage: {
		inputTokens: {
			total: tokens.input,
			noCache: tokens.input,
			cacheRead: undefined,
			cacheWrite: undefined,
		},
		outputTokens: { total: tokens.output, text: tokens.output, reasoning: undefined },
	},
});

const textStream = (text: string, tokens?: { input: number; output: number }) =>
	convertArrayToReadableStream<LanguageModelV3StreamPart>([
		{ type: "stream-start", warnings: [] },
		{ type: "text-start", id: "1" },
		{ type: "text-delta", id: "1", delta: text },
		{ type: "text-end", id: "1" },
		finishPart("stop", tokens),
	]);

// A stream that emits its text and then stays open until finish() or error().
function controlledTextStream(text: string) {
	let controller!: ReadableStreamDefaultController<LanguageModelV3StreamPart>;
	const stream = new ReadableStream<LanguageModelV3StreamPart>({
		start(c) {
			controller = c;
			c.enqueue({ type: "stream-start", warnings: [] });
			c.enqueue({ type: "text-start", id: "1" });
			c.enqueue({ type: "text-delta", id: "1", delta: text });
		},
	});
	return {
		stream,
		finish() {
			controller.enqueue({ type: "text-end", id: "1" });
			controller.enqueue(finishPart());
			controller.close();
		},
		error(error: unknown) {
			controller.enqueue({ type: "error", error });
			controller.close();
		},
	};
}

const hangingTextStream = (abortSignal: AbortSignal | undefined, text: string) =>
	new ReadableStream<LanguageModelV3StreamPart>({
		start(controller) {
			abortSignal?.addEventListener(
				"abort",
				() => controller.error(abortSignal.reason ?? new Error("aborted")),
				{ once: true },
			);
			controller.enqueue({ type: "stream-start", warnings: [] });
			controller.enqueue({ type: "text-start", id: "1" });
			controller.enqueue({ type: "text-delta", id: "1", delta: text });
		},
	});

const flat = (transcript: SessionTranscript) =>
	transcript.items.map((item) => `${item.kind === "user" ? "u" : "r"}:${item.text}`);

function responseItem(item: TranscriptItem | undefined): TranscriptResponseItem {
	if (item?.kind !== "response") throw new Error("expected a response item");
	return item;
}

function userItem(item: TranscriptItem | undefined): TranscriptUserItem {
	if (item?.kind !== "user") throw new Error("expected a user item");
	return item;
}

// ── textOf ──────────────────────────────────────────────────────────────

describe("textOf", () => {
	it("returns string content as-is", () => {
		expect(textOf({ role: "user", content: "plain text" })).toBe("plain text");
	});

	it("joins text parts and ignores non-text parts", () => {
		expect(
			textOf({
				role: "assistant",
				content: [
					{ type: "text", text: "The answer " },
					{ type: "tool-call", toolCallId: "t1", toolName: "lookup", input: {} },
					{ type: "text", text: "is 42." },
				],
			}),
		).toBe("The answer is 42.");
	});

	it("returns empty text for non-text content", () => {
		expect(
			textOf({
				role: "user",
				content: [{ type: "image", image: Uint8Array.of(1), mediaType: "image/png" }],
			}),
		).toBe("");
	});
});

// ── projectSession over hand-built ledgers ──────────────────────────────

describe("projectSession", () => {
	it("projects an empty session as idle with no items", () => {
		expect(projectSession([])).toEqual({ items: [], status: "idle", activeRunId: null });
	});

	it("distinguishes an interrupted open run from a live streaming one", () => {
		// The SIGKILL shape: run-start + a persisted step, no run-end.
		const events: StoredEvent[] = [
			user("t0", "u-a", "hi"),
			{ type: "run-start", at: "t1", runId: "r1", model: "m" },
			ackStep("t2", "r1", ["u-a"], "a-1", "half an answer"),
		];

		const dead = projectSession(events);
		expect(dead.status).toBe("interrupted");
		expect(dead.activeRunId).toBe("r1");
		expect(flat(dead)).toEqual(["u:hi", "r:half an answer"]);
		expect(dead.items[1]).toMatchObject({ id: "r1/0", status: "interrupted", at: "t2" });

		const alive = projectSession(events, { live: true });
		expect(alive.status).toBe("streaming");
		expect(alive.items[1]).toMatchObject({ id: "r1/0", status: "streaming" });
		expect(alive.items.map((item) => item.id)).toEqual(dead.items.map((item) => item.id));
	});

	it("projects an open run with no steps yet as an empty in-flight response", () => {
		const events: StoredEvent[] = [
			user("t0", "u-a", "hi"),
			{ type: "run-start", at: "t1", runId: "r1", model: "m" },
		];

		const live = projectSession(events, { live: true });
		expect(live.items.map((item) => item.kind)).toEqual(["user", "response"]);
		expect(live.items[0]).toMatchObject({ queued: false, runId: "r1" });
		expect(live.items[1]).toMatchObject({
			id: "r1/0",
			text: "",
			messages: [],
			status: "streaming",
			at: "t1",
		});
		expect(projectSession(events).items[1]).toMatchObject({ status: "interrupted" });

		// A run resumed for queued-only work has no initiating input at all.
		const bare = projectSession([{ type: "run-start", at: "t0", runId: "r2", model: "m" }], {
			live: true,
		});
		expect(bare.items).toEqual([
			expect.objectContaining({ kind: "response", id: "r2/0", text: "", status: "streaming" }),
		]);
	});

	it("keeps queued-only pending work idle with the queued item visible", () => {
		const events: StoredEvent[] = [
			user("t0", "u-a", "A"),
			{ type: "run-start", at: "t1", runId: "r1", model: "m" },
			ackStep("t2", "r1", ["u-a"], "a-1", "answer A"),
			{ type: "run-end", at: "t3", runId: "r1", status: "completed" },
			user("t4", "q-b", "B", { queued: true, queueId: "q-b", origin: "widget" }),
		];

		const transcript = projectSession(events);
		expect(transcript.status).toBe("idle");
		expect(transcript.activeRunId).toBeNull();
		expect(flat(transcript)).toEqual(["u:A", "r:answer A", "u:B"]);
		const queued = userItem(transcript.items[2]);
		expect(queued).toMatchObject({ queued: true, runId: null });
		// Framework queue markers are stripped; the app's own meta passes through.
		expect(queued.meta).toEqual({ origin: "widget" });
	});
});

// ── projections of ledgers produced by the real run loop ────────────────

describe("session transcripts over real run ledgers", () => {
	it("projects a completed single-turn chat as [user, response]", async () => {
		setContextWindow("mock/transcript-single", 100_000);
		const storage = memoryStorage();
		const model = new MockLanguageModelV3({
			doStream: async () => ({ stream: textStream("Hello!") }),
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const session = agentic.session("single", { model: "mock/transcript-single" });
		const result = await session.send("Hi", { meta: { topic: "greeting" } });

		const events = await storage.load("single");
		const transcript = projectSession(events);
		expect(transcript.status).toBe("idle");
		expect(transcript.activeRunId).toBeNull();
		expect(flat(transcript)).toEqual(["u:Hi", "r:Hello!"]);

		const input = userItem(transcript.items[0]);
		const answer = responseItem(transcript.items[1]);
		expect(input).toMatchObject({
			queued: false,
			runId: result.runId,
			meta: { topic: "greeting" },
		});
		expect(answer).toMatchObject({
			id: `${result.runId}/0`,
			runId: result.runId,
			status: "completed",
		});

		// Identities and times mirror the ledger exactly.
		const userEvent = events.find(
			(event): event is Extract<StoredEvent, { type: "user-message" }> =>
				event.type === "user-message",
		);
		const stepEvent = events.find(
			(event): event is Extract<StoredEvent, { type: "step" }> => event.type === "step",
		);
		expect(input.id).toBe(userEvent?.id);
		expect(input.at).toBe(userEvent?.at);
		expect(answer.at).toBe(stepEvent?.at);
		expect(answer.messages.map((entry) => entry.id)).toEqual(
			stepEvent?.messages.map((entry) => entry.id),
		);
	});

	it("projects a multi-turn session in order with per-run response ids", async () => {
		setContextWindow("mock/transcript-multi", 100_000);
		const storage = memoryStorage();
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				return { stream: textStream(calls === 1 ? "first answer" : "second answer") };
			},
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const session = agentic.session("multi", { model: "mock/transcript-multi" });
		const first = await session.send("first?");
		const second = await session.send("second?");

		const transcript = projectSession(await storage.load("multi"));
		expect(flat(transcript)).toEqual([
			"u:first?",
			"r:first answer",
			"u:second?",
			"r:second answer",
		]);
		expect(transcript.items.map((item) => item.runId)).toEqual([
			first.runId,
			first.runId,
			second.runId,
			second.runId,
		]);
		expect(transcript.items[1]?.id).toBe(`${first.runId}/0`);
		expect(transcript.items[3]?.id).toBe(`${second.runId}/0`);
		expect(transcript.status).toBe("idle");
	});

	it("splits a run's response around a message queued mid-run, claimed exactly once", async () => {
		setContextWindow("mock/transcript-queued-split", 100_000);
		const storage = memoryStorage();
		let calls = 0;
		let first!: ReturnType<typeof controlledTextStream>;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				if (calls === 1) {
					first = controlledTextStream("Answer A");
					return { stream: first.stream };
				}
				return { stream: textStream("Answer B") };
			},
		});
		let queuedDurably!: () => void;
		const queued = new Promise<void>((resolve) => {
			queuedDurably = resolve;
		});
		const agentic = createAgentic({
			storage,
			getModel: () => model,
			onEvent: (event) => {
				if (event.type === "queued-message") queuedDurably();
			},
		});
		const session = agentic.session("queued-split", { model: "mock/transcript-queued-split" });
		let sawDelta!: () => void;
		const firstDelta = new Promise<void>((resolve) => {
			sawDelta = resolve;
		});

		const firstSend = session.send("A", {
			onPart: (part) => {
				if (part.type === "text-delta") sawDelta();
			},
		});
		await firstDelta; // pass 1 snapshot is taken; B can only be folded in later
		const secondSend = session.send("B", { meta: { source: "followup" } });
		await queued;

		// Mid-run: the queued message is visible below the open response bubble.
		const mid = projectSession(await storage.load("queued-split"), { live: true });
		expect(mid.status).toBe("streaming");
		expect(flat(mid)).toEqual(["u:A", "r:", "u:B"]);
		expect(mid.items[1]).toMatchObject({ status: "streaming" });
		expect(mid.items[2]).toMatchObject({ queued: true, runId: null });

		first.finish();
		const [a, b] = await Promise.all([firstSend, secondSend]);
		expect(a.runId).toBe(b.runId);

		const transcript = projectSession(await storage.load("queued-split"));
		expect(flat(transcript)).toEqual(["u:A", "r:Answer A", "u:B", "r:Answer B"]);
		expect(transcript.items[1]).toMatchObject({ id: `${a.runId}/0`, status: "completed" });
		expect(transcript.items[3]).toMatchObject({ id: `${a.runId}/1`, status: "completed" });
		// The streaming bubble's id survived to the terminal projection.
		expect(mid.items[1]?.id).toBe(transcript.items[1]?.id);

		const followup = userItem(transcript.items[2]);
		expect(followup).toMatchObject({ queued: false, runId: a.runId });
		// Framework queue markers are stripped; the app meta stays.
		expect(followup.meta).toEqual({ source: "followup" });
		// Claimed exactly once.
		expect(
			transcript.items.filter((item) => item.kind === "user" && item.id === followup.id),
		).toHaveLength(1);
		expect(transcript.status).toBe("idle");
	});

	it("projects a cancelled run with its salvaged partial text", async () => {
		setContextWindow("mock/transcript-cancel", 100_000);
		const storage = memoryStorage();
		const abort = new AbortController();
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => ({
				stream: hangingTextStream(abortSignal, "kept partial answer"),
			}),
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const session = agentic.session("cancelled", { model: "mock/transcript-cancel" });

		const result = await session.send("stop me", {
			abortSignal: abort.signal,
			onPart: (part) => {
				if (part.type === "text-delta") abort.abort(new Error("user pressed stop"));
			},
		});
		expect(result.status).toBe("cancelled");

		const transcript = projectSession(await storage.load("cancelled"));
		expect(transcript.status).toBe("idle");
		expect(transcript.activeRunId).toBeNull();
		expect(flat(transcript)).toEqual(["u:stop me", "r:kept partial answer"]);
		expect(transcript.items[0]).toMatchObject({ queued: false, runId: result.runId });
		expect(transcript.items[1]).toMatchObject({ id: `${result.runId}/0`, status: "cancelled" });
	});

	it("keeps a failed run's queued input queued, then a successor claims it exactly once", async () => {
		setContextWindow("mock/transcript-failed-queue", 100_000);
		const storage = memoryStorage();
		const sessionId = "failed-queue";
		await storage.append(sessionId, user("t0", "u-a", "A"));
		let calls = 0;
		let first!: ReturnType<typeof controlledTextStream>;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				if (calls === 1) {
					first = controlledTextStream("doomed partial");
					return { stream: first.stream };
				}
				return { stream: textStream("Answer B") };
			},
		});
		let sawDelta!: () => void;
		const firstDelta = new Promise<void>((resolve) => {
			sawDelta = resolve;
		});
		const mailbox = createMailbox();
		const run = runLoop({
			sessionId,
			agent: { model: "mock/transcript-failed-queue" },
			storage,
			getModel: () => model,
			mailbox,
			onPart: (part) => {
				if (part.type === "text-delta") sawDelta();
			},
		});
		await firstDelta;
		await storage.append(sessionId, user("t1", "q-b", "B", { queued: true, queueId: "q-b" }));
		mailbox.tryEnqueue("q-b");
		first.error(new Error("provider exploded"));
		const failed = await run;
		expect(failed.status).toBe("failed");

		// The failure surfaces on the response (with the error step's salvaged
		// partial text); the unseen queued input stays queued.
		const afterFailure = projectSession(await storage.load(sessionId));
		expect(afterFailure.status).toBe("idle");
		expect(flat(afterFailure)).toEqual(["u:A", "r:doomed partial", "u:B"]);
		expect(afterFailure.items[1]).toMatchObject({
			id: `${failed.runId}/0`,
			status: "failed",
			error: { message: "provider exploded" },
		});
		expect(afterFailure.items[2]).toMatchObject({ queued: true, runId: null });

		const successor = await runLoop({
			sessionId,
			agent: { model: "mock/transcript-failed-queue" },
			storage,
			getModel: () => model,
		});
		expect(successor.status).toBe("completed");

		const transcript = projectSession(await storage.load(sessionId));
		expect(flat(transcript)).toEqual(["u:A", "r:doomed partial", "u:B", "r:Answer B"]);
		// Earlier items are untouched by the successor — only B's claim resolves.
		expect(transcript.items.slice(0, 2)).toEqual(afterFailure.items.slice(0, 2));
		expect(transcript.items[2]).toMatchObject({ id: "q-b", queued: false, runId: successor.runId });
		expect(transcript.items[3]).toMatchObject({ id: `${successor.runId}/0`, status: "completed" });
		expect(transcript.items.filter((item) => item.id === "q-b")).toHaveLength(1);
	});

	it("projects an identical transcript across real compaction", async () => {
		setContextWindow("mock/transcript-compaction", 100);
		const storage = memoryStorage();
		let calls = 0;
		const model = new MockLanguageModelV3({
			doGenerate: async () => ({
				content: [{ type: "text" as const, text: "summary of earlier history" }],
				finishReason: { unified: "stop" as const, raw: "stop" },
				usage: {
					inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
					outputTokens: { total: 5, text: 5, reasoning: undefined },
				},
				warnings: [],
			}),
			doStream: async () => {
				calls += 1;
				// Turn 1 reports context usage over the 50% threshold, forcing a
				// real post-run compaction; turn 2 stays under it.
				if (calls === 1) {
					return { stream: textStream("first answer", { input: 60, output: 20 }) };
				}
				return { stream: textStream("second answer") };
			},
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const session = agentic.session("compacted", {
			model: "mock/transcript-compaction",
			compaction: { limit: 0.5, keepRecent: 0 },
		});
		await session.send("first question");

		const events = await storage.load("compacted");
		const compactionIndex = events.findIndex((event) => event.type === "compaction");
		expect(compactionIndex).toBeGreaterThan(-1);

		const before = projectSession(events.slice(0, compactionIndex));
		const after = projectSession(events);
		expect(after).toEqual(before);

		const second = await session.send("second question");
		const full = projectSession(await storage.load("compacted"));
		expect(flat(full)).toEqual([
			"u:first question",
			"r:first answer",
			"u:second question",
			"r:second answer",
		]);
		// The pre-compaction transcript is a verbatim prefix; the summary is
		// a model-view artifact and never becomes an item.
		expect(full.items.slice(0, before.items.length)).toEqual(before.items);
		expect(full.items[3]?.id).toBe(`${second.runId}/0`);
		expect(JSON.stringify(full.items)).not.toContain("summary of earlier history");
	});

	it("hides poke messages by default and shows them with internal: true", async () => {
		setContextWindow("mock/transcript-poke", 100_000);
		const storage = memoryStorage();
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				if (calls === 1) {
					// Ends the turn without a terminal tool — gets poked.
					return { stream: textStream("Still thinking about it.") };
				}
				return {
					stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
						{ type: "stream-start", warnings: [] },
						{
							type: "tool-call",
							toolCallId: "submit-1",
							toolName: "submit_deliverable",
							input: JSON.stringify({ deliverable: "done" }),
						},
						finishPart("tool-calls"),
					]),
				};
			},
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const outcome = await agentic.task({
			id: "poked-task",
			agent: { model: "mock/transcript-poke" },
			prompt: "Deliver something.",
		});
		expect(outcome.status).toBe("submitted");

		const events = await storage.load("poked-task");
		expect(
			events.some((event) => event.type === "user-message" && event.meta?.poke !== undefined),
		).toBe(true);

		const transcript = projectSession(events);
		expect(transcript.items.map((item) => item.kind)).toEqual(["user", "response", "response"]);
		expect(userItem(transcript.items[0]).text).toBe("Deliver something.");
		expect(JSON.stringify(transcript.items)).not.toContain("ended your turn");

		const runId = responseItem(transcript.items[1]).runId;
		const debug = projectSession(events, { internal: true });
		expect(debug.items.map((item) => item.kind)).toEqual(["user", "response", "user", "response"]);
		// The poke sits between the response segments it split, flagged via meta.
		expect(debug.items[2]).toMatchObject({ queued: false, runId, meta: { poke: 1 } });
		expect(debug.items[1]?.id).toBe(`${runId}/0`);
		expect(debug.items[3]?.id).toBe(`${runId}/1`);
		expect(transcript.items[1]?.id).toBe(`${runId}/0`);
		expect(transcript.items[2]?.id).toBe(`${runId}/1`);
	});

	it("session.transcript() tracks live streaming through to the terminal state", async () => {
		setContextWindow("mock/transcript-live", 100_000);
		const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "agentic-transcript-"));
		tempDirs.push(dir);
		const storage = fileStorage(dir);
		let stream!: ReturnType<typeof controlledTextStream>;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				stream = controlledTextStream("Streaming answer");
				return { stream: stream.stream };
			},
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const session = agentic.session("live-transcript", { model: "mock/transcript-live" });
		await expect(session.transcript()).resolves.toEqual({
			items: [],
			status: "idle",
			activeRunId: null,
		});

		let sawDelta!: () => void;
		const firstDelta = new Promise<void>((resolve) => {
			sawDelta = resolve;
		});
		const pending = session.send("Question", {
			onPart: (part) => {
				if (part.type === "text-delta") sawDelta();
			},
		});
		await firstDelta;

		const live = await session.transcript();
		expect(live.status).toBe("streaming");
		expect(flat(live)).toEqual(["u:Question", "r:"]);
		expect(live.items[1]).toMatchObject({ status: "streaming", messages: [] });

		stream.finish();
		const result = await pending;
		expect(live.activeRunId).toBe(result.runId);
		expect(live.items[1]?.id).toBe(`${result.runId}/0`);

		const done = await session.transcript();
		expect(done.status).toBe("idle");
		expect(done.activeRunId).toBeNull();
		expect(flat(done)).toEqual(["u:Question", "r:Streaming answer"]);
		expect(done.items[1]).toMatchObject({ id: `${result.runId}/0`, status: "completed" });
	});
});
