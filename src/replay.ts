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
	/**
	 * User messages with no model response yet — queued messages a crash
	 * orphaned before (or during) their run. A run-end of any status clears
	 * them: completed answered them, cancelled/failed settled them
	 * deliberately (auto-resuming those would re-run an abort or replay a
	 * failure on every boot).
	 */
	pendingMessages: number;
	/**
	 * Auto-resume sweep attempts on the currently-open work (run-resume
	 * events with auto=true since the last run-end). A run-end resets the
	 * count. The sweep uses this as its crash-loop breaker: a run that keeps
	 * killing the process on resume stops being retried after the cap.
	 */
	autoResumeAttempts: number;
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
	let pendingMessages = 0;
	let autoResumeAttempts = 0;
	const openRuns = new Map<string, true>();

	for (const event of events) {
		switch (event.type) {
			case "user-message":
				messages.push(event.message);
				pendingMessages += 1;
				break;
			case "step":
				messages.push(...event.messages);
				totals = addStepToTotals(totals, event.usage);
				pendingMessages = 0;
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
			case "run-resume":
				if (event.auto) autoResumeAttempts += 1;
				break;
			case "run-end":
				openRuns.delete(event.runId);
				pendingMessages = 0;
				autoResumeAttempts = 0;
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
		pendingMessages,
		autoResumeAttempts,
		repaired: sanitized.removed,
	};
}
