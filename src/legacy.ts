import type { ModelMessage } from "ai";
import { encodeEvent, parseEventJson } from "./serialize.js";
import type { StoredEvent, StoredMessage } from "./types.js";

/**
 * One-time migration for ledgers written before agentic v0.7.
 *
 * v0.7 introduced message identity and v0.8 added the causal input membership
 * (`inputMessageIds`) that projection reads; `decodeEvent` refuses anything
 * older rather than guessing at it. This upgrades such a ledger in place —
 * once, offline — so the strict decoder stays the only runtime read path.
 *
 * Nothing here is invented. Legacy `step` events already recorded the two
 * fields the reconstruction needs (`acknowledgesInput` and `inputQueueIds`),
 * so membership is derived by replaying the session's own settle rule — the
 * same one `replaySession` applies — rather than inferred from append order.
 * Only identity is minted, and it is derived from ledger position, so running
 * the upgrade twice yields byte-identical output.
 */

function queueIdOf(meta: unknown): string | null {
	if (!meta || typeof meta !== "object") return null;
	const record = meta as Record<string, unknown>;
	const value = record.queueId;
	return record.queued === true && typeof value === "string" ? value : null;
}

function isStoredMessage(value: unknown): value is StoredMessage {
	return (
		!!value &&
		typeof value === "object" &&
		typeof (value as StoredMessage).id === "string" &&
		"message" in (value as StoredMessage)
	);
}

/** Wrap raw ModelMessages in identity envelopes; leave existing ones alone. */
function storedMessagesOf(raw: unknown, eventIndex: number): StoredMessage[] {
	if (!Array.isArray(raw)) return [];
	return raw.map((entry, position) =>
		isStoredMessage(entry)
			? entry
			: { id: `legacy:${eventIndex}:${position}`, message: entry as ModelMessage },
	);
}

interface PendingInput {
	id: string;
	queueId: string | null;
}

/**
 * Pre-v0.7 steps re-persisted the whole run each time: step 2 of a tool loop
 * repeated step 1's messages. A current step records only what it added, so
 * concatenating the old shape duplicates text. Detect it exactly — the step's
 * messages begin with everything the run has already recorded, and go further
 * — and keep only the tail. Anything else is taken as already incremental.
 */
function newMessagesOf(messages: StoredMessage[], emitted: string[]): StoredMessage[] {
	if (emitted.length === 0 || messages.length <= emitted.length) return messages;
	const repeatsRun = emitted.every(
		(seen, index) => JSON.stringify(messages[index]?.message) === seen,
	);
	return repeatsRun ? messages.slice(emitted.length) : messages;
}

/**
 * Upgrade one session's stored events, in append order, to the current schema.
 *
 * Pass every raw string the storage provider holds for the session — the walk
 * is stateful, so a partial slice would settle inputs against an incomplete
 * view. Events already in the current schema pass through unchanged (their
 * identity is never re-minted), which makes this safe for the mixed ledgers
 * left by an app that upgraded mid-conversation.
 *
 * Returns the re-encoded events, positionally aligned with the input, ready to
 * write back with the same provider.
 */
export function upgradeLegacySession(rawEvents: string[]): string[] {
	let pending: PendingInput[] = [];
	// Messages each run has already recorded, as JSON, for cumulative-step detection.
	const emittedByRun = new Map<string, string[]>();

	return rawEvents.map((raw, index) => {
		const event = parseEventJson(raw) as Record<string, unknown>;

		switch (event.type) {
			case "user-message": {
				if (typeof event.id !== "string") {
					// v0.6 called it `inputId`. It is already a stable unique value,
					// so adopting it keeps ids consistent with anything that recorded
					// them (queue markers, app-side references) before the rename.
					event.id = typeof event.inputId === "string" ? event.inputId : `legacy:${index}:input`;
				}
				delete event.inputId;
				pending.push({ id: event.id as string, queueId: queueIdOf(event.meta) });
				break;
			}

			case "step": {
				const runId = typeof event.runId === "string" ? event.runId : "";
				const emitted = emittedByRun.get(runId) ?? [];
				const messages = newMessagesOf(storedMessagesOf(event.messages, index), emitted);
				emittedByRun.set(runId, [
					...emitted,
					...messages.map((stored) => JSON.stringify(stored.message)),
				]);
				event.messages = messages;
				if (!Array.isArray(event.inputQueueIds)) event.inputQueueIds = [];
				if (typeof event.acknowledgesInput !== "boolean") event.acknowledgesInput = false;

				// Settle exactly what replaySession would settle for this step: a
				// non-queued input, or a queued one this step's own record names.
				const included = new Set(event.inputQueueIds as string[]);
				const settles = event.acknowledgesInput
					? pending.filter((input) => input.queueId === null || included.has(input.queueId))
					: [];
				if (!Array.isArray(event.inputMessageIds)) {
					event.inputMessageIds = settles.map((input) => input.id);
				}
				if (event.acknowledgesInput) {
					pending = pending.filter(
						(input) => input.queueId !== null && !included.has(input.queueId),
					);
				}
				break;
			}

			case "compaction": {
				event.messages = storedMessagesOf(event.messages, index);
				// v0.7 dropped this field; replay rebuilds pending from the events
				// themselves, which this walk is already tracking.
				delete event.pendingInputs;
				break;
			}
		}

		return encodeEvent(event as unknown as StoredEvent);
	});
}
