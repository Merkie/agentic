import type { ModelMessage } from "ai";
import { sanitizeConversation } from "./sanitize.js";
import type { StoredEvent, UsageTotals } from "./types.js";
import { addStepToTotals, emptyTotals } from "./usage.js";

export interface ReplayedSession {
	/** The replay-ready conversation (sanitized). */
	messages: ModelMessage[];
	/** Latest step's context size (input+output tokens) — drives compaction. */
	contextTokens: number | null;
	/** Lifetime totals across every run and compaction in the ledger. */
	totals: UsageTotals;
	/** runId of a run that started but never ended (crash/kill), if any. */
	interruptedRunId: string | null;
	/** How many malformed parts the sanitizer had to repair on load. */
	repaired: number;
}

/**
 * Rebuild session state from the event ledger. A `compaction` event resets
 * the message base (its `messages` supersede everything before it); every
 * `user-message` and `step` after the base appends in order. Totals span the
 * whole ledger — compaction never erases cost history.
 */
export function replaySession(events: StoredEvent[]): ReplayedSession {
	let messages: ModelMessage[] = [];
	let totals = emptyTotals();
	const openRuns = new Map<string, true>();

	for (const event of events) {
		switch (event.type) {
			case "user-message":
				messages.push(event.message);
				break;
			case "step":
				messages.push(...event.messages);
				totals = addStepToTotals(totals, event.usage);
				break;
			case "compaction":
				messages = [...event.messages];
				// The summarizer's cost counts toward lifetime totals, but its
				// context size is the OLD conversation being summarized — the
				// live context is unknown until the next step reports usage.
				if (event.usage) totals = addStepToTotals(totals, event.usage);
				totals = { ...totals, contextTokens: null };
				break;
			case "run-start":
				openRuns.set(event.runId, true);
				break;
			case "run-end":
				openRuns.delete(event.runId);
				break;
		}
	}

	const sanitized = sanitizeConversation(messages);
	const lastOpen = [...openRuns.keys()].pop() ?? null;

	return {
		messages: sanitized.messages,
		contextTokens: totals.contextTokens,
		totals,
		interruptedRunId: lastOpen,
		repaired: sanitized.removed,
	};
}
