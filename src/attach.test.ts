import type { LanguageModelV3StreamPart } from "@ai-sdk/provider";
import { type TextStreamPart, type ToolSet, tool } from "ai";
import { convertArrayToReadableStream, MockLanguageModelV3 } from "ai/test";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { createAgentic } from "./agentic.js";
import { setContextWindow } from "./modelMeta.js";
import { type ProgressEvent, progressFromPart } from "./progress.js";
import type { StreamContext, TranscriptResponseItem } from "./projection.js";
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
// BEFORE pushing, signal() from a part listener.
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

interface SeenPart {
	type: string;
	text?: string;
	context: StreamContext;
}

const seen = (part: TextStreamPart<ToolSet>, context: StreamContext): SeenPart => ({
	type: part.type,
	...(part.type === "text-delta" ? { text: part.text } : {}),
	context,
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
	it("attach mid-run delivers subsequent parts with context; detach stops delivery", async () => {
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

		const received: SeenPart[] = [];
		const detach = session.attach((part, context) => received.push(seen(part, context)));
		delivered = gate.next();
		stream.delta("B");
		await delivered;

		const deltas = received.filter((entry) => entry.type === "text-delta");
		expect(deltas).toEqual([
			{ type: "text-delta", text: "B", context: expect.objectContaining({ offset: 1 }) },
		]); // the earlier "A" is not replayed

		detach();
		delivered = gate.next();
		stream.delta("C");
		await delivered;
		expect(received.filter((entry) => entry.type === "text-delta")).toHaveLength(1);

		stream.finish();
		const result = await pending;
		expect(result.text).toBe("ABC");
		expect(deltas[0]?.context).toMatchObject({
			runId: result.runId,
			responseId: `${result.runId}/0`,
		});
	});

	it("a listener attached before any run receives parts from runs that start later", async () => {
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

		const received: SeenPart[] = [];
		session.attach((part, context) => received.push(seen(part, context)));

		const first = await session.send("one");
		const second = await session.send("two");
		const deltas = received.filter((entry) => entry.type === "text-delta");
		expect(deltas.map((entry) => entry.text)).toEqual(["first", "second"]);
		expect(deltas.map((entry) => entry.context.responseId)).toEqual([
			`${first.runId}/0`,
			`${second.runId}/0`,
		]);
	});

	it("receives queued-pickup passes too — attach is session-scoped, not run-scoped", async () => {
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

		const received: SeenPart[] = [];
		session.attach((part, context) => received.push(seen(part, context)));
		const secondSend = session.send("B");
		await queued;
		first.finish();
		const [a, b] = await Promise.all([firstSend, secondSend]);
		expect(b.runId).toBe(a.runId);

		// Segment 0's delta streamed before the attach; the queued pickup's pass
		// (segment 1 of the SAME run) was delivered to the attached listener.
		const deltas = received.filter((entry) => entry.type === "text-delta");
		expect(deltas).toEqual([
			{
				type: "text-delta",
				text: "Answer B",
				context: expect.objectContaining({ responseId: `${a.runId}/1`, runId: a.runId }),
			},
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
		const received: SeenPart[] = [];
		session.attach((part, context) => received.push(seen(part, context)));
		const callerDeltas: string[] = [];

		const result = await session.send("Q", {
			onPart: (part) => {
				if (part.type === "text-delta") callerDeltas.push(part.text);
			},
		});
		expect(result.status).toBe("completed");
		expect(callerDeltas).toEqual(["Hello!"]);
		expect(received.some((entry) => entry.type === "text-delta")).toBe(true);
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

		const received: SeenPart[] = [];
		agentic
			.session("tailed-task", { model: "mock/attach-task" })
			.attach((part, context) => received.push(seen(part, context)));

		const outcome = await agentic.task({
			id: "tailed-task",
			durable: false,
			agent: { model: "mock/attach-task" },
			prompt: "Deliver something.",
		});
		expect(outcome.status).toBe("submitted");
		// The isolated in-memory ledger changes where events persist, not whether
		// the session broadcasts: attach is keyed by session id at delivery time.
		expect(received.map((entry) => entry.type)).toContain("tool-call");
		expect(received.map((entry) => entry.type)).toContain("tool-result");
		expect(await storage.load("tailed-task")).toEqual([]);
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

	it("partialText length always equals the next delta's offset (reconnect dedupe invariant)", async () => {
		setContextWindow("mock/partial-offset", 100_000);
		const storage = memoryStorage();
		const stream = steppedModel();
		const agentic = createAgentic({ storage, getModel: () => stream.model });
		const session = agentic.session("partial-offset", { model: "mock/partial-offset" });

		const gate = partGate();
		const buffered: SeenPart[] = [];
		session.attach((part, context) => {
			buffered.push(seen(part, context));
			if (part.type === "text-delta") gate.signal();
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

		const bufferedDeltas = buffered.filter((entry) => entry.type === "text-delta");
		expect(bufferedDeltas.map((entry) => entry.context.offset)).toEqual([0, 2, 5]);
		// Offsets only exist on text deltas.
		for (const entry of buffered) {
			if (entry.type !== "text-delta") expect(entry.context.offset).toBeUndefined();
		}
		// The invariant: whatever the interleaving, a snapshot's partialText
		// length IS the offset of the next delta...
		for (const [index, snapshot] of snapshots.entries()) {
			const next = bufferedDeltas[index + 1];
			if (next) expect(snapshot.length).toBe(next.context.offset);
		}
		// ...so the reconnect recipe (drop deltas with offset < partialText
		// length, append the rest) rebuilds the exact text from ANY snapshot.
		for (const snapshot of snapshots) {
			const rebuilt =
				snapshot +
				bufferedDeltas
					.filter((entry) => (entry.context.offset ?? 0) >= snapshot.length)
					.map((entry) => entry.text)
					.join("");
			expect(rebuilt).toBe("abcdefg");
		}
	});
});

// ── progressFromPart ────────────────────────────────────────────────────

describe("progressFromPart", () => {
	const context: StreamContext = { runId: "r1", responseId: "r1/0" };
	const part = (value: Record<string, unknown>) => value as unknown as TextStreamPart<ToolSet>;

	it("maps text deltas with offset passthrough", () => {
		expect(
			progressFromPart(part({ type: "text-delta", id: "1", text: "hi" }), {
				...context,
				offset: 7,
			}),
		).toEqual({ type: "text", responseId: "r1/0", runId: "r1", delta: "hi", offset: 7 });
		// No offset on the context (direct runLoop use) degrades to 0.
		expect(progressFromPart(part({ type: "text-delta", id: "1", text: "hi" }), context)).toEqual({
			type: "text",
			responseId: "r1/0",
			runId: "r1",
			delta: "hi",
			offset: 0,
		});
	});

	it("maps reasoning deltas", () => {
		expect(
			progressFromPart(part({ type: "reasoning-delta", id: "1", text: "hmm" }), context),
		).toEqual({ type: "reasoning", responseId: "r1/0", runId: "r1", delta: "hmm" });
	});

	it("maps tool-call to tool-start and authoritative tool-result to tool-end", () => {
		expect(
			progressFromPart(
				part({ type: "tool-call", toolCallId: "c1", toolName: "search", input: {} }),
				context,
			),
		).toEqual({
			type: "tool-start",
			responseId: "r1/0",
			runId: "r1",
			toolCallId: "c1",
			toolName: "search",
		});
		expect(
			progressFromPart(
				part({ type: "tool-result", toolCallId: "c1", toolName: "search", input: {}, output: 1 }),
				context,
			),
		).toEqual({
			type: "tool-end",
			responseId: "r1/0",
			runId: "r1",
			toolCallId: "c1",
			toolName: "search",
		});
		// A preliminary result is a preview — the tool is still running.
		expect(
			progressFromPart(
				part({
					type: "tool-result",
					toolCallId: "c1",
					toolName: "search",
					input: {},
					output: 1,
					preliminary: true,
				}),
				context,
			),
		).toBeNull();
	});

	it("returns null for lifecycle and bookkeeping parts", () => {
		const uninteresting = [
			"start",
			"start-step",
			"finish-step",
			"finish",
			"text-start",
			"text-end",
			"reasoning-start",
			"reasoning-end",
			"tool-input-start", // tool-start comes from tool-call — see the JSDoc
			"tool-input-delta",
			"tool-input-end",
			"tool-error",
			"abort",
			"error",
			"raw",
		];
		for (const type of uninteresting) {
			expect(progressFromPart(part({ type }), context)).toBeNull();
		}
	});

	it("yields [tool-start, tool-end, text] with consistent ids over a real tool run", async () => {
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
		const result = await session.send("Hi", {
			onPart: (streamPart, streamContext) => {
				const event = progressFromPart(streamPart, streamContext);
				if (event) events.push(event);
			},
		});
		expect(result.status).toBe("completed");
		expect(events).toEqual([
			{
				type: "tool-start",
				responseId: `${result.runId}/0`,
				runId: result.runId,
				toolCallId: "t-1",
				toolName: "touch",
			},
			{
				type: "tool-end",
				responseId: `${result.runId}/0`,
				runId: result.runId,
				toolCallId: "t-1",
				toolName: "touch",
			},
			{
				type: "text",
				responseId: `${result.runId}/0`,
				runId: result.runId,
				delta: "Done.",
				offset: 0,
			},
		]);
		// Transport-ready: the events survive JSON serialization verbatim.
		expect(JSON.parse(JSON.stringify(events))).toEqual(events);
	});
});
