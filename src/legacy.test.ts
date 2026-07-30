import { describe, expect, it } from "vitest";
import { upgradeLegacySession } from "./legacy.js";
import { projectSession } from "./projection.js";
import { replaySession } from "./replay.js";
import { decodeEvent, encodeEvent } from "./serialize.js";
import type { StoredEvent } from "./types.js";

const USAGE = {
	inputTokens: 1,
	outputTokens: 1,
	totalTokens: 2,
	cachedInputTokens: 0,
	reasoningTokens: 0,
	cost: 0,
	upstreamCost: 0,
	billedCost: 0,
	isByok: false,
};

/** A pre-v0.7 ledger: `inputId` instead of `id`, raw messages, no membership. */
function legacyTurn(options: {
	runId: string;
	inputId: string;
	text: string;
	reply: string;
	at?: string;
}): string[] {
	const at = options.at ?? "2026-07-13T04:02:14.637Z";
	return [
		JSON.stringify({
			type: "user-message",
			at,
			message: { role: "user", content: options.text },
			inputId: options.inputId,
			meta: { source: "test" },
		}),
		JSON.stringify({ type: "run-start", at, runId: options.runId, model: "mock/model" }),
		JSON.stringify({
			type: "step",
			at,
			runId: options.runId,
			messages: [{ role: "assistant", content: [{ type: "text", text: options.reply }] }],
			inputQueueIds: [],
			acknowledgesInput: true,
			finishReason: "stop",
			usage: USAGE,
		}),
		JSON.stringify({ type: "run-end", at, runId: options.runId, status: "completed" }),
	];
}

const decodeAll = (raw: string[]): StoredEvent[] => raw.map((entry) => decodeEvent(entry));

describe("upgradeLegacySession", () => {
	it("makes a pre-v0.7 ledger decodable, and renders it", () => {
		const raw = legacyTurn({ runId: "run-1", inputId: "input-1", text: "yo", reply: "hey" });
		expect(() => decodeAll(raw)).toThrow(/before agentic v0.7/);

		const upgraded = upgradeLegacySession(raw);
		const transcript = projectSession(decodeAll(upgraded));

		expect(transcript.items.map((item) => [item.kind, item.text])).toEqual([
			["user", "yo"],
			["response", "hey"],
		]);
		expect(transcript.items[0]?.id).toBe("input-1"); // v0.6 inputId adopted as identity
		expect(transcript.status).toBe("idle");
	});

	it("settles the input so recovery does not re-drive finished conversations", () => {
		const upgraded = upgradeLegacySession(
			legacyTurn({ runId: "run-1", inputId: "input-1", text: "yo", reply: "hey" }),
		);
		const replayed = replaySession(decodeAll(upgraded));

		expect(replayed.pendingMessages).toBe(0);
		expect(replayed.interruptedRunId).toBeNull();
	});

	it("derives membership from the step's own record, not append order", () => {
		// A queued input the step did NOT see must stay pending, exactly as the
		// step's recorded inputQueueIds says — never swept in by position.
		const raw = [
			JSON.stringify({
				type: "user-message",
				at: "2026-07-13T04:00:00.000Z",
				message: { role: "user", content: "first" },
				inputId: "input-1",
			}),
			JSON.stringify({
				type: "run-start",
				at: "2026-07-13T04:00:01.000Z",
				runId: "run-1",
				model: "mock/model",
			}),
			JSON.stringify({
				type: "user-message",
				at: "2026-07-13T04:00:02.000Z",
				message: { role: "user", content: "queued mid-run" },
				inputId: "input-2",
				meta: { queued: true, queueId: "q-2" },
			}),
			JSON.stringify({
				type: "step",
				at: "2026-07-13T04:00:03.000Z",
				runId: "run-1",
				messages: [{ role: "assistant", content: [{ type: "text", text: "answer" }] }],
				inputQueueIds: [], // did not see q-2
				acknowledgesInput: true,
				finishReason: "stop",
				usage: USAGE,
			}),
		];

		const events = decodeAll(upgradeLegacySession(raw));
		const step = events.find((event) => event.type === "step");
		expect(step?.type === "step" && step.inputMessageIds).toEqual(["input-1"]);

		// The unseen queued input survives as pending work.
		expect(replaySession(events).pendingQueueIds).toEqual(["q-2"]);
	});

	it("de-duplicates cumulative tool-loop steps", () => {
		// Pre-v0.7 re-persisted the whole run on every step. Concatenated
		// verbatim that repeats the earlier passes' text in the transcript.
		const assistant = (text: string) => ({
			role: "assistant",
			content: [{ type: "text", text }],
		});
		const tool = { role: "tool", content: [] };
		const step = (at: string, messages: unknown[]) =>
			JSON.stringify({
				type: "step",
				at,
				runId: "run-1",
				messages,
				inputQueueIds: [],
				acknowledgesInput: true,
				finishReason: "tool-calls",
				usage: USAGE,
			});

		const raw = [
			JSON.stringify({
				type: "user-message",
				at: "2026-07-13T04:00:00.000Z",
				message: { role: "user", content: "look it up" },
				inputId: "input-1",
			}),
			JSON.stringify({
				type: "run-start",
				at: "2026-07-13T04:00:01.000Z",
				runId: "run-1",
				model: "mock/model",
			}),
			step("2026-07-13T04:00:02.000Z", [assistant("Looking it up."), tool]),
			step("2026-07-13T04:00:03.000Z", [assistant("Looking it up."), tool, assistant("Found it.")]),
			JSON.stringify({
				type: "run-end",
				at: "2026-07-13T04:00:04.000Z",
				runId: "run-1",
				status: "completed",
			}),
		];

		const transcript = projectSession(decodeAll(upgradeLegacySession(raw)));
		const response = transcript.items.find((item) => item.kind === "response");
		expect(response?.text).toBe("Looking it up.\n\nFound it.");
		expect(response?.text).not.toContain("Looking it up.\n\nLooking it up.");
	});

	it("leaves current-schema events untouched and is idempotent", () => {
		const current = encodeEvent({
			type: "user-message",
			at: "2026-07-28T15:51:11.000Z",
			message: { role: "user", content: "modern" },
			id: "real-uuid",
			meta: { source: "test" },
		} as StoredEvent);

		const once = upgradeLegacySession([current]);
		expect(decodeEvent(once[0] as string)).toMatchObject({ id: "real-uuid" });
		expect(upgradeLegacySession(once)).toEqual(once);
	});

	it("upgrades a ledger that switched schemas mid-conversation", () => {
		const legacy = legacyTurn({
			runId: "run-1",
			inputId: "input-1",
			text: "old",
			reply: "old reply",
		});
		const modern = [
			encodeEvent({
				type: "user-message",
				at: "2026-07-28T00:00:00.000Z",
				message: { role: "user", content: "new" },
				id: "input-2",
			} as StoredEvent),
			encodeEvent({
				type: "run-start",
				at: "2026-07-28T00:00:01.000Z",
				runId: "run-2",
				model: "mock/model",
			} as StoredEvent),
			encodeEvent({
				type: "step",
				at: "2026-07-28T00:00:02.000Z",
				runId: "run-2",
				messages: [
					{
						id: "m-1",
						message: { role: "assistant", content: [{ type: "text", text: "new reply" }] },
					},
				],
				inputMessageIds: ["input-2"],
				inputQueueIds: [],
				acknowledgesInput: true,
				finishReason: "stop",
				usage: USAGE,
			} as StoredEvent),
			encodeEvent({
				type: "run-end",
				at: "2026-07-28T00:00:03.000Z",
				runId: "run-2",
				status: "completed",
			} as StoredEvent),
		];

		const transcript = projectSession(decodeAll(upgradeLegacySession([...legacy, ...modern])));
		expect(transcript.items.map((item) => item.text)).toEqual([
			"old",
			"old reply",
			"new",
			"new reply",
		]);
	});

	it("preserves binary message parts through the re-encode", () => {
		const bytes = new Uint8Array([1, 2, 3, 250]);
		const raw = [
			JSON.stringify({
				type: "user-message",
				at: "2026-07-13T04:00:00.000Z",
				message: { role: "user", content: "look" },
				inputId: "input-1",
			}),
			encodeEvent({
				type: "compaction",
				at: "2026-07-13T04:00:01.000Z",
				messages: [{ role: "user", content: [{ type: "image", image: bytes }] }],
			} as unknown as StoredEvent),
		];

		const events = decodeAll(upgradeLegacySession(raw));
		const compaction = events.find((event) => event.type === "compaction");
		const part = compaction?.type === "compaction" ? compaction.messages[0] : undefined;
		const content = part?.message.content as Array<{ image: Uint8Array }>;
		expect(content[0]?.image).toBeInstanceOf(Uint8Array);
		expect([...(content[0]?.image ?? [])]).toEqual([1, 2, 3, 250]);
	});
});
