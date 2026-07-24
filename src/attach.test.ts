import type { LanguageModelV3StreamPart } from "@ai-sdk/provider";
import { tool } from "ai";
import { convertArrayToReadableStream, MockLanguageModelV3 } from "ai/test";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { createAgentic } from "./agentic.js";
import { setContextWindow } from "./modelMeta.js";
import type { ProgressEvent } from "./progress.js";
import { type TranscriptResponseItem, wireTranscript } from "./projection.js";
import { memoryStorage } from "./storage.js";

// ── stream fixtures (see transcript.test.ts for the shared patterns) ────

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

const textStream = (text: string) =>
	convertArrayToReadableStream<LanguageModelV3StreamPart>([
		{ type: "stream-start", warnings: [] },
		{ type: "text-start", id: "1" },
		{ type: "text-delta", id: "1", delta: text },
		{ type: "text-end", id: "1" },
		finishPart(),
	]);

// A stream that emits its text and then stays open until finish().
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
	};
}

// A stream that emits its text and then hangs until the run is aborted.
const hangingTextStream = (abortSignal: AbortSignal | undefined, text: string) =>
	new ReadableStream<LanguageModelV3StreamPart>({
		start(controller) {
			const abort = () => controller.error(abortSignal?.reason ?? new Error("aborted"));
			abortSignal?.addEventListener("abort", abort, { once: true });
			controller.enqueue({ type: "stream-start", warnings: [] });
			controller.enqueue({ type: "text-start", id: "1" });
			controller.enqueue({ type: "text-delta", id: "1", delta: text });
		},
	});

// A stream whose text deltas are pushed one by one from the test.
function steppedTextStream() {
	let controller!: ReadableStreamDefaultController<LanguageModelV3StreamPart>;
	const stream = new ReadableStream<LanguageModelV3StreamPart>({
		start(c) {
			controller = c;
			c.enqueue({ type: "stream-start", warnings: [] });
			c.enqueue({ type: "text-start", id: "1" });
		},
	});
	return {
		stream,
		delta(text: string) {
			controller.enqueue({ type: "text-delta", id: "1", delta: text });
		},
		finish() {
			controller.enqueue({ type: "text-end", id: "1" });
			controller.enqueue(finishPart());
			controller.close();
		},
	};
}

// A mock model whose single stepped stream the test drives; streaming()
// resolves once doStream has handed the run its stream.
function steppedModel() {
	let stream!: ReturnType<typeof steppedTextStream>;
	let ready!: () => void;
	const streaming = new Promise<void>((resolve) => {
		ready = resolve;
	});
	const model = new MockLanguageModelV3({
		doStream: async () => {
			stream = steppedTextStream();
			ready();
			return { stream: stream.stream };
		},
	});
	return {
		model,
		streaming,
		delta: (text: string) => stream.delta(text),
		finish: () => stream.finish(),
	};
}

// Await the delivery of stream parts the test just pushed: register next()
// BEFORE pushing, signal() from a listener.
function partGate() {
	const waiters: Array<() => void> = [];
	return {
		signal() {
			waiters.shift()?.();
		},
		next() {
			return new Promise<void>((resolve) => {
				waiters.push(resolve);
			});
		},
	};
}

const textEvents = (events: ProgressEvent[]) =>
	events.filter((event): event is Extract<ProgressEvent, { type: "text" }> => {
		return event.type === "text";
	});

function streamingItem(items: Array<{ kind: string; id: string }>, id: string) {
	const item = items.find(
		(candidate) => candidate.kind === "response" && candidate.id === id,
	) as TranscriptResponseItem;
	if (!item) throw new Error(`no response item ${id}`);
	return item;
}

// ── session.attach ──────────────────────────────────────────────────────

describe("session.attach", () => {
	it("attach mid-run delivers subsequent progress; detach stops delivery", async () => {
		setContextWindow("mock/attach-mid", 100_000);
		const storage = memoryStorage();
		const stream = steppedModel();
		const agentic = createAgentic({ storage, getModel: () => stream.model });
		const session = agentic.session("attach-mid", { model: "mock/attach-mid" });

		const gate = partGate();
		const pending = session.send("Q", {
			onPart: (part) => {
				if (part.type === "text-delta") gate.signal();
			},
		});
		await stream.streaming;
		let delivered = gate.next();
		stream.delta("A");
		await delivered; // the pre-attach delta is fully delivered before attaching

		const received: ProgressEvent[] = [];
		const detach = session.attach((event) => received.push(event));
		delivered = gate.next();
		stream.delta("B");
		await delivered;

		const result = await (async () => {
			// the earlier "A" is not replayed, and the pass's response-start
			// happened before the attach — only the live "B" delta arrives
			expect(received).toHaveLength(1);

			detach();
			delivered = gate.next();
			stream.delta("C");
			await delivered;
			expect(received).toHaveLength(1);

			stream.finish();
			return pending;
		})();
		expect(result.text).toBe("ABC");
		// run-end was emitted after detach and did not arrive either
		expect(received).toEqual([
			{
				type: "text",
				runId: result.runId,
				responseId: `${result.runId}/0`,
				delta: "B",
				offset: 1,
			},
		]);
	});

	it("a listener attached before any run receives full sequences from runs that start later", async () => {
		setContextWindow("mock/attach-early", 100_000);
		const storage = memoryStorage();
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				return { stream: textStream(calls === 1 ? "first" : "second") };
			},
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const session = agentic.session("attach-early", { model: "mock/attach-early" });

		const received: ProgressEvent[] = [];
		session.attach((event) => received.push(event));

		const first = await session.send("one");
		const second = await session.send("two");
		// Both runs arrive completely: pass boundary, deltas, terminal marker —
		// including run-end for completed runs.
		expect(received).toEqual([
			{ type: "response-start", runId: first.runId, responseId: `${first.runId}/0` },
			{
				type: "text",
				runId: first.runId,
				responseId: `${first.runId}/0`,
				delta: "first",
				offset: 0,
			},
			{ type: "run-end", runId: first.runId, status: "completed" },
			{ type: "response-start", runId: second.runId, responseId: `${second.runId}/0` },
			{
				type: "text",
				runId: second.runId,
				responseId: `${second.runId}/0`,
				delta: "second",
				offset: 0,
			},
			{ type: "run-end", runId: second.runId, status: "completed" },
		]);
	});

	it("receives queued-pickup passes with their own response-start — attach is session-scoped, not run-scoped", async () => {
		setContextWindow("mock/attach-queued", 100_000);
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
		const session = agentic.session("attach-queued", { model: "mock/attach-queued" });

		let sawDelta!: () => void;
		const firstDelta = new Promise<void>((resolve) => {
			sawDelta = resolve;
		});
		const firstSend = session.send("A", {
			onPart: (part) => {
				if (part.type === "text-delta") sawDelta();
			},
		});
		await firstDelta;

		const received: ProgressEvent[] = [];
		session.attach((event) => received.push(event));
		const secondSend = session.send("B");
		await queued;
		first.finish();
		const [a, b] = await Promise.all([firstSend, secondSend]);
		expect(b.runId).toBe(a.runId);

		// Segment 0's delta streamed before the attach; the queued pickup's pass
		// (segment 1 of the SAME run) announced its boundary and streamed to the
		// attached listener, offsets restarting at 0.
		expect(received).toEqual([
			{ type: "response-start", runId: a.runId, responseId: `${a.runId}/1` },
			{ type: "text", runId: a.runId, responseId: `${a.runId}/1`, delta: "Answer B", offset: 0 },
			{ type: "run-end", runId: a.runId, status: "completed" },
		]);
	});

	it("contains listener errors; the run and other listeners are unaffected", async () => {
		setContextWindow("mock/attach-throw", 100_000);
		const storage = memoryStorage();
		const model = new MockLanguageModelV3({
			doStream: async () => ({ stream: textStream("Hello!") }),
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const session = agentic.session("attach-throw", { model: "mock/attach-throw" });

		session.attach(() => {
			throw new Error("broken listener");
		});
		const received: ProgressEvent[] = [];
		session.attach((event) => received.push(event));
		const callerDeltas: string[] = [];

		const result = await session.send("Q", {
			onPart: (part) => {
				if (part.type === "text-delta") callerDeltas.push(part.text);
			},
		});
		expect(result.status).toBe("completed");
		expect(callerDeltas).toEqual(["Hello!"]);
		expect(textEvents(received).map((event) => event.delta)).toEqual(["Hello!"]);
	});

	it("broadcasts task() runs, including durable: false ones", async () => {
		setContextWindow("mock/attach-task", 100_000);
		const storage = memoryStorage();
		const model = new MockLanguageModelV3({
			doStream: async () => ({
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
			}),
		});
		const agentic = createAgentic({ storage, getModel: () => model });

		const received: ProgressEvent[] = [];
		agentic
			.session("tailed-task", { model: "mock/attach-task" })
			.attach((event) => received.push(event));

		const outcome = await agentic.task({
			id: "tailed-task",
			durable: false,
			agent: { model: "mock/attach-task" },
			prompt: "Deliver something.",
		});
		expect(outcome.status).toBe("submitted");
		// The isolated in-memory ledger changes where events persist, not whether
		// the session broadcasts: attach is keyed by session id at delivery time.
		// tool-start carries the parsed tool input.
		expect(received.map((event) => event.type)).toEqual([
			"response-start",
			"tool-start",
			"tool-end",
			"run-end",
		]);
		expect(received[1]).toMatchObject({
			type: "tool-start",
			toolCallId: "submit-1",
			toolName: "submit_deliverable",
			input: { deliverable: "done" },
		});
		expect(received[3]).toMatchObject({ type: "run-end", status: "completed" });
		expect(await storage.load("tailed-task")).toEqual([]);
	});

	it("run-end reaches attach listeners for cancelled runs", async () => {
		setContextWindow("mock/attach-cancel", 100_000);
		const storage = memoryStorage();
		const model = new MockLanguageModelV3({
			doStream: async ({ abortSignal }) => ({
				stream: hangingTextStream(abortSignal, "kept partial"),
			}),
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const session = agentic.session("attach-cancel", { model: "mock/attach-cancel" });

		const received: ProgressEvent[] = [];
		session.attach((event) => received.push(event));

		let sawDelta!: () => void;
		const firstDelta = new Promise<void>((resolve) => {
			sawDelta = resolve;
		});
		const pending = session.send("Q", {
			onPart: (part) => {
				if (part.type === "text-delta") sawDelta();
			},
		});
		await firstDelta;
		expect(agentic.cancel("attach-cancel", new Error("stopped"))).toBe(true);
		const result = await pending;
		expect(result.status).toBe("cancelled");
		expect(received.at(-1)).toEqual({
			type: "run-end",
			runId: result.runId,
			status: "cancelled",
		});
	});
});

// ── in-flight partial text ──────────────────────────────────────────────

describe("transcript partialText", () => {
	it("overlays the text-so-far while streaming and drops it once durable", async () => {
		setContextWindow("mock/partial-text", 100_000);
		const storage = memoryStorage();
		const stream = steppedModel();
		const agentic = createAgentic({ storage, getModel: () => stream.model });
		const session = agentic.session("partial-text", { model: "mock/partial-text" });

		const gate = partGate();
		const pending = session.send("Q", {
			onPart: (part) => {
				if (part.type === "text-delta") gate.signal();
			},
		});
		await stream.streaming;
		let delivered = gate.next();
		stream.delta("Hel");
		await delivered;

		const runId = (await session.transcript()).activeRunId;
		const mid = await session.transcript();
		expect(mid.status).toBe("streaming");
		const midItem = streamingItem(mid.items, `${runId}/0`);
		expect(midItem).toMatchObject({ status: "streaming", text: "", partialText: "Hel" });

		delivered = gate.next();
		stream.delta("lo");
		await delivered;
		expect(streamingItem((await session.transcript()).items, `${runId}/0`).partialText).toBe(
			"Hello",
		);

		stream.finish();
		const result = await pending;
		const done = await session.transcript();
		const item = streamingItem(done.items, `${result.runId}/0`);
		expect(item).toMatchObject({ status: "completed", text: "Hello" });
		expect("partialText" in item).toBe(false);
	});

	it("partialText length always equals the next text event's offset (reconnect dedupe invariant)", async () => {
		setContextWindow("mock/partial-offset", 100_000);
		const storage = memoryStorage();
		const stream = steppedModel();
		const agentic = createAgentic({ storage, getModel: () => stream.model });
		const session = agentic.session("partial-offset", { model: "mock/partial-offset" });

		const gate = partGate();
		const buffered: ProgressEvent[] = [];
		session.attach((event) => {
			buffered.push(event);
			if (event.type === "text") gate.signal();
		});

		const pending = session.send("Q");
		await stream.streaming;
		const deltas = ["ab", "cde", "fg"];
		// Take a transcript snapshot at EVERY point between deltas.
		const snapshots: string[] = [];
		for (const text of deltas) {
			const delivered = gate.next();
			stream.delta(text);
			await delivered;
			const transcript = await session.transcript();
			const item = streamingItem(transcript.items, `${transcript.activeRunId}/0`);
			snapshots.push(item.partialText ?? "");
		}
		stream.finish();
		const result = await pending;
		expect(result.text).toBe("abcdefg");

		const bufferedTexts = textEvents(buffered);
		expect(bufferedTexts.map((event) => event.offset)).toEqual([0, 2, 5]);
		// The pass announced itself before its first delta.
		expect(buffered[0]).toEqual({
			type: "response-start",
			runId: result.runId,
			responseId: `${result.runId}/0`,
		});
		// The invariant: whatever the interleaving, a snapshot's partialText
		// length IS the offset of the next delta...
		for (const [index, snapshot] of snapshots.entries()) {
			const next = bufferedTexts[index + 1];
			if (next) expect(snapshot.length).toBe(next.offset);
		}
		// ...so the reconnect recipe (drop text events with offset < partialText
		// length, append the rest) rebuilds the exact text from ANY snapshot.
		for (const snapshot of snapshots) {
			const rebuilt =
				snapshot +
				bufferedTexts
					.filter((event) => event.offset >= snapshot.length)
					.map((event) => event.delta)
					.join("");
			expect(rebuilt).toBe("abcdefg");
		}
	});
});

// ── the progress pipeline ───────────────────────────────────────────────

describe("progress pipeline", () => {
	it("emits the exact sequence over a real tool run, ids consistent throughout", async () => {
		setContextWindow("mock/progress-run", 100_000);
		const storage = memoryStorage();
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				if (calls === 1) {
					return {
						stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
							{ type: "stream-start", warnings: [] },
							{ type: "tool-call", toolCallId: "t-1", toolName: "touch", input: "{}" },
							finishPart("tool-calls"),
						]),
					};
				}
				return { stream: textStream("Done.") };
			},
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const session = agentic.session("progress-run", {
			model: "mock/progress-run",
			tools: { touch: tool({ inputSchema: z.object({}), execute: async () => "touched" }) },
		});

		const events: ProgressEvent[] = [];
		const result = await session.send("Hi", { onProgress: (event) => events.push(event) });
		expect(result.status).toBe("completed");
		const responseId = `${result.runId}/0`;
		expect(events).toEqual([
			// pass 1: the tool call
			{ type: "response-start", runId: result.runId, responseId },
			{
				type: "tool-start",
				runId: result.runId,
				responseId,
				toolCallId: "t-1",
				toolName: "touch",
				input: {},
			},
			{ type: "tool-end", runId: result.runId, responseId, toolCallId: "t-1", toolName: "touch" },
			// pass 2 continues the same segment: same responseId, offsets reset
			{ type: "response-start", runId: result.runId, responseId },
			{ type: "text", runId: result.runId, responseId, delta: "Done.", offset: 0 },
			{ type: "run-end", runId: result.runId, status: "completed" },
		]);
		// Transport-ready: the events survive JSON serialization verbatim.
		expect(JSON.parse(JSON.stringify(events))).toEqual(events);
	});

	it("re-announces the SAME responseId on a retry pass, and retry carries the attempt", async () => {
		setContextWindow("mock/progress-retry", 100_000);
		const storage = memoryStorage();
		let calls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				calls += 1;
				if (calls === 1) {
					return {
						stream: convertArrayToReadableStream<LanguageModelV3StreamPart>([
							{ type: "stream-start", warnings: [] },
							{ type: "error", error: new Error("502 upstream idle timeout exceeded") },
						]),
					};
				}
				return { stream: textStream("ok") };
			},
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const session = agentic.session("progress-retry", {
			model: "mock/progress-retry",
			retry: { maxAttempts: 2, baseDelayMs: 0 },
		});

		const events: ProgressEvent[] = [];
		const result = await session.send("Q", { onProgress: (event) => events.push(event) });
		expect(result.status).toBe("completed");
		const responseId = `${result.runId}/0`;
		expect(events).toEqual([
			{ type: "response-start", runId: result.runId, responseId },
			{ type: "retry", runId: result.runId, attempt: 1 },
			// same responseId — the retry re-streams the same transcript item
			{ type: "response-start", runId: result.runId, responseId },
			{ type: "text", runId: result.runId, responseId, delta: "ok", offset: 0 },
			{ type: "run-end", runId: result.runId, status: "completed" },
		]);
	});

	it("SendOptions.onProgress delivers every event and is error-contained", async () => {
		setContextWindow("mock/progress-send", 100_000);
		const storage = memoryStorage();
		const model = new MockLanguageModelV3({
			doStream: async () => ({ stream: textStream("Hello!") }),
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		const session = agentic.session("progress-send", { model: "mock/progress-send" });

		const types: string[] = [];
		const result = await session.send("Q", {
			onProgress: (event) => {
				types.push(event.type);
				throw new Error("broken progress listener"); // contained on every event
			},
		});
		expect(result.status).toBe("completed");
		expect(types).toEqual(["response-start", "text", "run-end"]);
	});

	it("TaskOptions.onProgress delivers tool activity and is error-contained", async () => {
		setContextWindow("mock/progress-task", 100_000);
		const storage = memoryStorage();
		const model = new MockLanguageModelV3({
			doStream: async () => ({
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
			}),
		});
		const agentic = createAgentic({ storage, getModel: () => model });

		const events: ProgressEvent[] = [];
		const outcome = await agentic.task({
			durable: false,
			agent: { model: "mock/progress-task" },
			prompt: "Deliver something.",
			onProgress: (event) => {
				events.push(event);
				throw new Error("broken progress listener"); // contained on every event
			},
		});
		expect(outcome.status).toBe("submitted");
		expect(events.map((event) => event.type)).toEqual([
			"response-start",
			"tool-start",
			"tool-end",
			"run-end",
		]);
		expect(events[1]).toMatchObject({
			type: "tool-start",
			toolName: "submit_deliverable",
			input: { deliverable: "done" },
		});
	});
});

// ── config-free reads ───────────────────────────────────────────────────

describe("config-free reads", () => {
	it("agentic.transcript(sessionId) equals session.transcript(), live overlay included", async () => {
		setContextWindow("mock/read-transcript", 100_000);
		const storage = memoryStorage();
		const stream = steppedModel();
		const agentic = createAgentic({ storage, getModel: () => stream.model });
		const session = agentic.session("read-transcript", { model: "mock/read-transcript" });

		const gate = partGate();
		const pending = session.send("Q", {
			onPart: (part) => {
				if (part.type === "text-delta") gate.signal();
			},
		});
		await stream.streaming;
		const delivered = gate.next();
		stream.delta("Hel");
		await delivered;

		// Mid-stream: same projection, same partialText overlay — no AgentConfig.
		const midConfigFree = await agentic.transcript("read-transcript");
		expect(midConfigFree).toEqual(await session.transcript());
		expect(midConfigFree.status).toBe("streaming");
		const midItem = streamingItem(midConfigFree.items, `${midConfigFree.activeRunId}/0`);
		expect(midItem.partialText).toBe("Hel");

		stream.finish();
		await pending;
		expect(await agentic.transcript("read-transcript")).toEqual(await session.transcript());
		expect(await agentic.transcript("read-transcript", { internal: true })).toEqual(
			await session.transcript({ internal: true }),
		);
	});

	it("wireTranscript strips exactly message/messages and nothing else", async () => {
		setContextWindow("mock/wire", 100_000);
		const storage = memoryStorage();
		const model = new MockLanguageModelV3({
			doStream: async () => ({ stream: textStream("Hello!") }),
		});
		const agentic = createAgentic({ storage, getModel: () => model });
		await agentic.session("wire", { model: "mock/wire" }).send("Q");

		const transcript = await agentic.transcript("wire");
		expect(transcript.items.length).toBeGreaterThanOrEqual(2); // user + response
		const wire = wireTranscript(transcript);

		expect(wire.status).toBe(transcript.status);
		expect(wire.activeRunId).toBe(transcript.activeRunId);
		expect(wire.items).toHaveLength(transcript.items.length);
		for (const [index, item] of transcript.items.entries()) {
			const wireItem = wire.items[index] as Record<string, unknown>;
			if (item.kind === "user") {
				const { message: _message, ...rest } = item;
				expect("message" in wireItem).toBe(false);
				expect(wireItem).toEqual(rest); // the remainder is untouched
			} else {
				const { messages: _messages, ...rest } = item;
				expect("messages" in wireItem).toBe(false);
				expect(wireItem).toEqual(rest);
			}
		}
		// Pure: the source transcript keeps its model payloads.
		expect(transcript.items.every((item) => "message" in item || "messages" in item)).toBe(true);
		// The wire payload survives JSON round-tripping verbatim.
		expect(JSON.parse(JSON.stringify(wire))).toEqual(wire);
	});
});
