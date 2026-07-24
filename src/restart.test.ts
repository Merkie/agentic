// Restart-on-interrupt semantics (v0.9 Chunk G): a `run-end { discarded: true }`
// voids a dead run's conversational contribution — replay rewinds its step
// output and returns its inputs to pending — and `autoResume.onInterrupted`
// lets an app choose "resume" | "restart" | "fail" per session.

import type { LanguageModelV3StreamPart } from "@ai-sdk/provider";
import type { ModelMessage } from "ai";
import { convertArrayToReadableStream, MockLanguageModelV3 } from "ai/test";
import { describe, expect, it } from "vitest";
import { createAgentic } from "./agentic.js";
import { setContextWindow } from "./modelMeta.js";
import { projectSession } from "./projection.js";
import { replaySession } from "./replay.js";
import { decodeEvent, encodeEvent } from "./serialize.js";
import { memoryStorage } from "./storage.js";
import type { StepUsage, StorageProvider, StoredEvent, StoredMessage } from "./types.js";

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

const step = (
	at: string,
	runId: string,
	inputMessageIds: string[],
	inputQueueIds: string[],
	...messages: ModelMessage[]
): StoredEvent => ({
	type: "step",
	at,
	runId,
	messages: stored(...messages),
	inputMessageIds,
	inputQueueIds,
	acknowledgesInput: true,
	finishReason: "stop",
	usage: usage(),
});

const contents = (replayed: ReturnType<typeof replaySession>) =>
	replayed.messages.map((entry) => entry.message.content);

// ── replay: the snapshot-rewind ─────────────────────────────────────────

describe("replaySession discarded rewind", () => {
	// One interrupted-run ledger, closed two ways: `discarded` must rewind,
	// a plain failed close must keep today's behavior exactly.
	const interrupted: StoredEvent[] = [
		user("t0", "u-old", "old question"),
		{ type: "run-start", at: "t1", runId: "r0", model: "m" },
		step("t2", "r0", ["u-old"], [], { role: "assistant", content: "old answer" }),
		{ type: "run-end", at: "t3", runId: "r0", status: "completed" },
		user("t4", "u-1", "build it"),
		{ type: "run-start", at: "t5", runId: "r1", model: "m" },
		user("t6", "q-a", "also add a footer", { queued: true, queueId: "q-a" }),
		user("t7", "q-b", "and a favicon", { queued: true, queueId: "q-b" }),
		// The dead run answered u-1 and q-a (q-b arrived too late to be seen).
		step("t8", "r1", ["u-1", "q-a"], ["q-a"], { role: "assistant", content: "partial work" }),
	];

	it("rewinds step output and restores acknowledged inputs to pending in arrival order", () => {
		const replayed = replaySession([
			...interrupted,
			{
				type: "run-end",
				at: "t9",
				runId: "r1",
				status: "failed",
				discarded: true,
				error: { message: "Interrupted; restarting" },
			},
		]);

		// The discarded run's step output is gone; every message that arrived
		// during the run is back, in arrival order.
		expect(contents(replayed)).toEqual([
			"old question",
			"old answer",
			"build it",
			"also add a footer",
			"and a favicon",
		]);
		// ALL of the run's inputs are pending again — the discarded run's
		// acknowledgments (u-1, q-a) are void.
		expect(replayed.pendingMessages).toBe(3);
		expect(replayed.pendingInputMessages.map((entry) => entry.id)).toEqual(["u-1", "q-a", "q-b"]);
		expect(replayed.pendingQueueIds).toEqual(["q-a", "q-b"]);
		// The run is closed (recoverable via the queued path, not resume-in-place)…
		expect(replayed.interruptedRunId).toBeNull();
		// …and usage totals are NOT rewound: the discarded step's cost was real.
		expect(replayed.totals.steps).toBe(2);
		expect(replayed.totals.cost).toBeCloseTo(0.002);
		// The live conversation size is unknown until the fresh run's first step.
		expect(replayed.contextTokens).toBeNull();
	});

	it("regression pair: a plain failed run-end keeps today's non-rewind behavior", () => {
		const replayed = replaySession([
			...interrupted,
			{ type: "run-end", at: "t9", runId: "r1", status: "failed" },
		]);

		// Step output survives; the acknowledging step causally hoisted u-1 and
		// q-a to sit under its output (unseen q-b stays where it arrived); the
		// failed close settles the ordinary initiating input and the step
		// settled q-a, so only unseen queued q-b stays pending.
		expect(contents(replayed)).toEqual([
			"old question",
			"old answer",
			"and a favicon",
			"build it",
			"also add a footer",
			"partial work",
		]);
		expect(replayed.pendingMessages).toBe(1);
		expect(replayed.pendingQueueIds).toEqual(["q-b"]);
		expect(replayed.interruptedRunId).toBeNull();
		expect(replayed.contextTokens).toBe(120);
	});

	it("does not restore pokes appended during the discarded run", () => {
		const replayed = replaySession([
			user("t0", "u-1", "do the task"),
			{ type: "run-start", at: "t1", runId: "r1", model: "m" },
			step("t2", "r1", ["u-1"], [], { role: "assistant", content: "ended turn idly" }),
			user("t3", "p-1", "You must call a terminal tool now.", { poke: 1 }),
			{
				type: "run-end",
				at: "t4",
				runId: "r1",
				status: "failed",
				discarded: true,
				error: { message: "Interrupted; restarting" },
			},
		]);

		// The poke's purpose (prodding THIS run to settle) died with the run —
		// restoring it would nag the fresh run with a stale instruction.
		expect(contents(replayed)).toEqual(["do the task"]);
		expect(replayed.pendingInputMessages.map((entry) => entry.id)).toEqual(["u-1"]);
	});

	it("mid-run compaction vanishes with the rewind; restored refs stay coherent for a successor", () => {
		const beforeRestart: StoredEvent[] = [
			user("t0", "u-old", "old question"),
			{ type: "run-start", at: "t1", runId: "r0", model: "m" },
			step("t2", "r0", ["u-old"], [], { role: "assistant", content: "old answer" }),
			{ type: "run-end", at: "t3", runId: "r0", status: "completed" },
			user("t4", "u-new", "current question"),
			{ type: "run-start", at: "t5", runId: "r1", model: "m" },
			step("t6", "r1", ["u-new"], [], { role: "assistant", content: "working on it" }),
			// Mid-run compaction rebases everything onto a summary. The pending
			// re-link machinery runs here; the rewind must still restore the
			// ORIGINAL pre-run refs, not the rebased ones.
			{
				type: "compaction",
				at: "t7",
				messages: [
					{ id: "sum-1", message: { role: "user", content: "<summary>" } },
					{ id: "u-new", at: "t4", message: { role: "user", content: "current question" } },
				],
				usage: usage({ cost: 0.005, billedCost: 0.005 }),
			},
			step("t8", "r1", [], [], { role: "assistant", content: "more work after compaction" }),
			{
				type: "run-end",
				at: "t9",
				runId: "r1",
				status: "failed",
				discarded: true,
				error: { message: "Interrupted; restarting" },
			},
		];

		const replayed = replaySession(beforeRestart);
		// As if the run never ran: no summary, no step content, pre-run history intact.
		expect(contents(replayed)).toEqual(["old question", "old answer", "current question"]);
		expect(JSON.stringify(replayed.messages)).not.toContain("<summary>");
		expect(replayed.pendingInputMessages.map((entry) => entry.id)).toEqual(["u-new"]);
		// The summarizer's cost still counts — totals span the whole ledger.
		expect(replayed.totals.cost).toBeCloseTo(0.001 + 0.001 + 0.001 + 0.005);

		// A successor run answers the restored input: acknowledgment must settle
		// it against the restored refs without duplicating the message.
		const afterRestart = replaySession([
			...beforeRestart,
			{ type: "run-start", at: "t10", runId: "r2", model: "m" },
			step("t11", "r2", ["u-new"], [], { role: "assistant", content: "fresh answer" }),
			{ type: "run-end", at: "t12", runId: "r2", status: "completed" },
		]);
		expect(contents(afterRestart)).toEqual([
			"old question",
			"old answer",
			"current question",
			"fresh answer",
		]);
		expect(afterRestart.pendingMessages).toBe(0);
	});

	it("accumulates auto-resume attempts across discarded closes (the restart crash-loop breaker)", () => {
		const replayed = replaySession([
			user("t0", "u-1", "build it"),
			{ type: "run-start", at: "t1", runId: "r-dead", model: "m" },
			{
				type: "run-end",
				at: "t2",
				runId: "r-dead",
				status: "failed",
				discarded: true,
				error: { message: "Interrupted; restarting" },
			},
			{ type: "run-resume", at: "t3", runId: null, auto: true },
			{ type: "run-start", at: "t4", runId: "rf-1", model: "m" },
			{
				type: "run-end",
				at: "t5",
				runId: "rf-1",
				status: "failed",
				discarded: true,
				error: { message: "Interrupted; restarting" },
			},
			{ type: "run-resume", at: "t6", runId: null, auto: true },
			{ type: "run-start", at: "t7", runId: "rf-2", model: "m" },
		]);

		// Each discarded close restores the input to pending, so run-end never
		// resets the breaker mid-loop; the count survives across restart cycles.
		expect(replayed.autoResumeAttempts).toBe(2);
		expect(replayed.interruptedRunId).toBe("rf-2");
		expect(replayed.pendingMessages).toBe(1);
	});
});

// ── projection: verify, don't change ────────────────────────────────────

describe("projectSession over a discarded + restarted ledger", () => {
	it("shows [user, failed response, successor response] with the user claimed once", () => {
		const events: StoredEvent[] = [
			user("t0", "u-1", "build the landing page"),
			{ type: "run-start", at: "t1", runId: "r1", model: "m" },
			step("t2", "r1", ["u-1"], [], { role: "assistant", content: "partial work" }),
			{
				type: "run-end",
				at: "t3",
				runId: "r1",
				status: "failed",
				discarded: true,
				error: { message: "Interrupted; restarting" },
			},
			{ type: "run-resume", at: "t4", runId: null, auto: true },
			{ type: "run-start", at: "t5", runId: "r2", model: "m" },
			step("t6", "r2", ["u-1"], [], { role: "assistant", content: "rebuilt and finished" }),
			{ type: "run-end", at: "t7", runId: "r2", status: "completed" },
		];

		const transcript = projectSession(events);
		expect(transcript.status).toBe("idle");
		expect(transcript.items.map((item) => item.kind)).toEqual(["user", "response", "response"]);
		// First claim wins: the discarded run keeps the user message; it is
		// neither duplicated before the successor nor stuck queued.
		expect(transcript.items[0]).toMatchObject({
			kind: "user",
			id: "u-1",
			runId: "r1",
			queued: false,
		});
		expect(transcript.items[1]).toMatchObject({
			kind: "response",
			id: "r1/0",
			status: "failed",
			text: "partial work",
			error: { message: "Interrupted; restarting" },
		});
		expect(transcript.items[2]).toMatchObject({
			kind: "response",
			id: "r2/0",
			status: "completed",
			text: "rebuilt and finished",
		});
	});
});

// ── codec: additive field round-trip ────────────────────────────────────

describe("decodeEvent", () => {
	it("round-trips a discarded run-end", () => {
		const event: StoredEvent = {
			type: "run-end",
			at: "t0",
			runId: "r1",
			status: "failed",
			discarded: true,
			error: { message: "Interrupted; restarting" },
		};
		expect(decodeEvent(encodeEvent(event))).toEqual({ ...event, v: 1 });

		// Absent stays absent — never materialized as false.
		const plain = decodeEvent(
			encodeEvent({ type: "run-end", at: "t0", runId: "r1", status: "failed" }),
		);
		expect("discarded" in plain).toBe(false);
	});
});

// ── the sweep: onInterrupted verdicts end to end ────────────────────────

describe("auto-resume onInterrupted", () => {
	const textStream = (text: string, finishReason: "stop" | "tool-calls" = "stop") =>
		convertArrayToReadableStream<LanguageModelV3StreamPart>([
			{ type: "stream-start", warnings: [] },
			{ type: "text-start", id: "1" },
			{ type: "text-delta", id: "1", delta: text },
			{ type: "text-end", id: "1" },
			{
				type: "finish",
				finishReason: { unified: finishReason, raw: finishReason },
				usage: {
					inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
					outputTokens: { total: 5, text: 5, reasoning: undefined },
				},
			},
		]);

	const drainBootSweep = () => new Promise((resolve) => setTimeout(resolve, 0));

	async function waitUntil(predicate: () => boolean | Promise<boolean>): Promise<void> {
		for (let attempt = 0; attempt < 100; attempt++) {
			if (await predicate()) return;
			await new Promise((resolve) => setTimeout(resolve, 0));
		}
		throw new Error("condition was not reached");
	}

	// Simulates a SIGKILL at the run-end write: the event is dropped while the
	// caller believes it landed, leaving the exact crash shape a dead process
	// leaves (run-start + steps, no run-end).
	function crashingStorage(
		inner: StorageProvider,
		dropRunEnd: (event: Extract<StoredEvent, { type: "run-end" }>) => boolean,
	): StorageProvider {
		return {
			append(sessionId, event) {
				if (event.type === "run-end" && dropRunEnd(event)) return;
				return inner.append(sessionId, event);
			},
			load: (sessionId) => inner.load(sessionId),
			listSessions: async () => (await inner.listSessions?.()) ?? [],
		};
	}

	it("restart: discards the dead run and re-drives its input in a fresh run", async () => {
		const modelId = "mock/restart-e2e";
		setContextWindow(modelId, 100_000);
		const sid = "astro-build";
		const inner = memoryStorage();
		let crashed = false;
		const storage = crashingStorage(inner, () => {
			if (crashed) return false;
			crashed = true;
			return true;
		});

		// Generation 1: a REAL run whose run-end write "dies with the process".
		// The persisted step ends on "tool-calls" — interrupted MID-WORK, so a
		// fresh process cannot reconcile it as a lost-marker completion.
		const model1 = new MockLanguageModelV3({
			doStream: async () => ({ stream: textStream("partial sandbox work", "tool-calls") }),
		});
		const agentic1 = createAgentic({ storage, getModel: () => model1 });
		await agentic1.session(sid, { model: modelId }).send("build the landing page");
		const dead = replaySession(await inner.load(sid));
		expect(dead.interruptedRunId).not.toBeNull();
		const deadRunId = dead.interruptedRunId as string;

		// Generation 2: a fresh process. The bash-tool sandbox died with the old
		// one, so the app restarts instead of resuming in place.
		const prompts: unknown[] = [];
		const model2 = new MockLanguageModelV3({
			doStream: async (options) => {
				prompts.push(options.prompt);
				return { stream: textStream("rebuilt and finished") };
			},
		});
		const hookSessions: string[] = [];
		const agentic2 = createAgentic({
			storage,
			getModel: () => model2,
			autoResume: {
				agentFor: () => ({ model: modelId }),
				onInterrupted: (sessionId) => {
					hookSessions.push(sessionId);
					return "restart";
				},
			},
		});
		// The boot sweep and this call race benignly (kicks re-check under the
		// session lock, so exactly one restarts the session); wait for the
		// winner's fresh run to close before asserting.
		const results = await agentic2.resumeInterrupted();
		await waitUntil(async () =>
			(await inner.load(sid)).some(
				(event) => event.type === "run-end" && event.status === "completed",
			),
		);
		expect(results.every((result) => result.status === "completed")).toBe(true);
		expect(hookSessions).toEqual([sid]);

		// The fresh run's model input excludes the dead run's step output but
		// includes the original user message.
		expect(prompts).toHaveLength(1);
		const promptText = JSON.stringify(prompts[0]);
		expect(promptText).toContain("build the landing page");
		expect(promptText).not.toContain("partial sandbox work");

		// Ledger: discarded close of the dead run, then a queued-style fresh run.
		const events = await inner.load(sid);
		expect(events.filter((event) => event.type === "run-end")).toEqual([
			expect.objectContaining({ runId: deadRunId, status: "failed", discarded: true }),
			expect.objectContaining({ status: "completed" }),
		]);
		expect(events.filter((event) => event.type === "run-resume")).toEqual([
			expect.objectContaining({ runId: null, auto: true }),
		]);
		expect(await agentic2.interruptedSessions()).toEqual([]);

		// Transcript: [user, failed (discarded) response, completed response],
		// the user item claimed exactly once — by the dead run.
		const transcript = await agentic2.transcript(sid);
		expect(
			transcript.items.map((item) => ({
				kind: item.kind,
				...(item.kind === "response" ? { status: item.status } : {}),
			})),
		).toEqual([
			{ kind: "user" },
			{ kind: "response", status: "failed" },
			{ kind: "response", status: "completed" },
		]);
		expect(transcript.items[0]).toMatchObject({ runId: deadRunId, queued: false });
		expect(transcript.items[2]).toMatchObject({ text: "rebuilt and finished" });
	});

	it("gives up after maxAttempts when every restart crashes, leaving the discarded runs", async () => {
		const modelId = "mock/restart-crash-loop";
		setContextWindow(modelId, 100_000);
		const sid = "astro-crash-loop";
		const inner = memoryStorage();
		// Every fresh run fails fatally and its run-end write "dies with the
		// process" — the astro shape of a build that kills the server each time.
		const storage = crashingStorage(
			inner,
			(event) => event.error?.message?.includes("kaboom") === true,
		);
		let modelCalls = 0;
		const model = new MockLanguageModelV3({
			doStream: async () => {
				modelCalls += 1;
				throw new Error("Invalid API key — kaboom");
			},
		});
		let hookCalls = 0;
		const giveUps: string[] = [];
		const agentic = createAgentic({
			storage,
			getModel: () => model,
			autoResume: {
				agentFor: () => ({ model: modelId }),
				maxAttempts: 2,
				onInterrupted: () => {
					hookCalls += 1;
					return "restart";
				},
			},
			onEvent: (event) => {
				if (event.type === "auto-resume" && event.action === "give-up")
					giveUps.push(event.sessionId);
			},
		});
		await drainBootSweep();
		await storage.append(sid, user("t0", "u-1", "build it"));
		await storage.append(sid, { type: "run-start", at: "t1", runId: "r-dead", model: modelId });

		// Two sweeps restart (and crash) twice; the third trips the ledger-
		// counted cap BEFORE consulting the hook and closes the run plainly.
		await agentic.resumeInterrupted();
		await agentic.resumeInterrupted();
		await agentic.resumeInterrupted();

		expect(modelCalls).toBe(2);
		expect(hookCalls).toBe(2);
		expect(giveUps).toEqual([sid]);

		const events = await inner.load(sid);
		const ends = events.filter(
			(event): event is Extract<StoredEvent, { type: "run-end" }> => event.type === "run-end",
		);
		// The failed discarded generations stay in the ledger; the give-up close
		// is a plain failure, NOT discarded.
		expect(ends.map((event) => event.discarded)).toEqual([true, true, undefined]);
		expect(ends[0]).toMatchObject({ runId: "r-dead", status: "failed" });
		expect(ends[2].error?.message).toContain("attempt limit exhausted");
		expect(
			events.filter((event) => event.type === "run-resume" && event.auto === true),
		).toHaveLength(2);
		// Closed, not wedged: the session leaves the recovery radar.
		expect(await agentic.interruptedSessions()).toEqual([]);
	});

	it("fail: closes the run with no re-drive, no attempt, and unseen queued input preserved", async () => {
		const modelId = "mock/restart-fail-verb";
		setContextWindow(modelId, 100_000);
		const sid = "astro-fail";
		const storage = memoryStorage();
		const runEnds: string[] = [];
		const agentic = createAgentic({
			storage,
			getModel: () => {
				throw new Error("model should not run");
			},
			autoResume: {
				agentFor: () => ({ model: modelId }),
				onInterrupted: () => "fail",
			},
			onEvent: (event) => {
				if (event.type === "run-end") runEnds.push(event.status);
			},
		});
		await drainBootSweep();
		await storage.append(sid, user("t0", "u-1", "build it"));
		await storage.append(sid, { type: "run-start", at: "t1", runId: "r1", model: modelId });
		await storage.append(sid, user("t2", "q-x", "unseen extra", { queued: true, queueId: "q-x" }));

		await expect(agentic.resumeInterrupted()).resolves.toEqual([]);

		const events = await storage.load(sid);
		expect(events.filter((event) => event.type === "run-end")).toEqual([
			expect.objectContaining({
				runId: "r1",
				status: "failed",
				error: expect.objectContaining({ message: expect.stringContaining("onInterrupted") }),
			}),
		]);
		expect(events.some((event) => event.type === "run-end" && event.discarded)).toBe(false);
		// No re-drive and no attempt consumed — mirror of the give-up path.
		expect(events.filter((event) => event.type === "run-resume")).toEqual([]);
		expect(runEnds).toEqual(["failed"]);

		const replayed = replaySession(events);
		expect(replayed.interruptedRunId).toBeNull();
		expect(replayed.autoResumeAttempts).toBe(0);
		// The initiating input is settled; the unseen queued input survives for
		// a later sweep or manual resume().
		expect(replayed.pendingQueueIds).toEqual(["q-x"]);
		expect(replayed.pendingMessages).toBe(1);
		expect(await agentic.interruptedSessions()).toEqual([sid]);
	});

	it.each([
		["absent", undefined],
		["explicit resume", (): "resume" => "resume"],
		[
			"throwing",
			(): never => {
				throw new Error("broken hook");
			},
		],
		["rejecting", (): Promise<never> => Promise.reject(new Error("broken hook"))],
	] as const)("%s hook keeps resume-in-place behavior unchanged", async (_label, onInterrupted) => {
		const modelId = "mock/restart-resume-default";
		setContextWindow(modelId, 100_000);
		const sid = `resume-anchor-${_label.replace(/ /g, "-")}`;
		const storage = memoryStorage();
		await storage.append(sid, user("t0", "u-1", "count to three"));
		await storage.append(sid, { type: "run-start", at: "t1", runId: "r1", model: modelId });
		await storage.append(sid, {
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
		const agentic = createAgentic({
			storage,
			getModel: () => model,
			autoResume: {
				agentFor: () => ({ model: modelId }),
				...(onInterrupted ? { onInterrupted } : {}),
			},
		});
		await agentic.resumeInterrupted();
		await waitUntil(async () =>
			(await storage.load(sid)).some((event) => event.type === "run-end"),
		);

		// Resumed IN PLACE under the original runId — the anchor behavior.
		const events = await storage.load(sid);
		expect(events.filter((event) => event.type === "run-resume")).toEqual([
			expect.objectContaining({ runId: "r1", auto: true }),
		]);
		expect(events.filter((event) => event.type === "run-start")).toHaveLength(1);
		expect(events.filter((event) => event.type === "run-end")).toEqual([
			expect.objectContaining({ runId: "r1", status: "completed" }),
		]);
		expect(events.some((event) => event.type === "run-end" && event.discarded)).toBe(false);
		expect(await agentic.interruptedSessions()).toEqual([]);
	});
});
