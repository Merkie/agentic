import type { ModelMessage } from "ai";
import { sanitizeConversation } from "./sanitize.js";
import type { StoredEvent, UsageTotals } from "./types.js";
import { addStepToTotals, emptyTotals } from "./usage.js";

function moveInputsToTail(
	messages: ModelMessage[],
	inputs: Array<{ message: ModelMessage }>,
): void {
	const tail: ModelMessage[] = [];
	for (const input of inputs) {
		const index = messages.indexOf(input.message);
		if (index >= 0) messages.splice(index, 1);
		tail.push(input.message);
	}
	messages.push(...tail);
}

export interface ReplayedSession {
	/** The replay-ready conversation (sanitized). */
	messages: ModelMessage[];
	/** Latest step's context size (input+output tokens) — drives compaction. */
	contextTokens: number | null;
	/** Lifetime totals across every run and compaction in the ledger. */
	totals: UsageTotals;
	/** runId of a run that started but never ended (crash/kill), if any. */
	interruptedRunId: string | null;
	/** Queued user-message ids not yet included in a persisted model step. */
	pendingQueueIds: string[];
	/** Exact unanswered input messages, in arrival order, for causal prompt hoisting. */
	pendingInputMessages: ModelMessage[];
	/**
	 * User messages with no model response yet — queued messages a crash
	 * orphaned before (or during) their run. Ordinary inputs are settled by a
	 * run-end of any status, but accepted queued inputs remain pending until a
	 * step explicitly records that they were part of its model input.
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
	let pendingInputs: Array<{ queueId: string | null; message: ModelMessage }> = [];
	let autoResumeAttempts = 0;
	const openRuns = new Map<string, true>();

	for (const event of events) {
		switch (event.type) {
			case "user-message": {
				messages.push(event.message);
				const queueId = event.meta?.queueId;
				pendingInputs.push({
					queueId: event.meta?.queued === true && typeof queueId === "string" ? queueId : null,
					message: event.message,
				});
				break;
			}
			case "step": {
				const acknowledgesInput = event.acknowledgesInput !== false;
				if (acknowledgesInput && event.inputQueueIds !== undefined) {
					const included = new Set(event.inputQueueIds);
					moveInputsToTail(
						messages,
						pendingInputs.filter((input) => input.queueId === null || included.has(input.queueId)),
					);
				}
				messages.push(...event.messages);
				totals = addStepToTotals(totals, event.usage);
				if (!acknowledgesInput) break;
				if (event.inputQueueIds === undefined) {
					// Legacy steps had no causal membership. Preserve their historical
					// sequence-based behavior when replaying an old ledger.
					pendingInputs = [];
				} else {
					const included = new Set(event.inputQueueIds);
					pendingInputs = pendingInputs.filter(
						(input) => input.queueId !== null && !included.has(input.queueId),
					);
				}
				break;
			}
			case "compaction": {
				messages = [...event.messages];
				for (const preserved of event.pendingInputs ?? []) {
					const input = pendingInputs[preserved.pendingIndex];
					const rebasedMessage = messages[preserved.messageIndex];
					if (input && rebasedMessage) input.message = rebasedMessage;
				}
				// The summarizer's cost counts toward lifetime totals, but its
				// context size is the OLD conversation being summarized — the
				// live context is unknown until the next step reports usage.
				if (event.usage) totals = addStepToTotals(totals, event.usage);
				totals = { ...totals, contextTokens: null };
				break;
			}
			case "run-start":
				openRuns.set(event.runId, true);
				break;
			case "run-resume":
				if (event.auto) autoResumeAttempts += 1;
				break;
			case "run-end":
				openRuns.delete(event.runId);
				// A cancelled/failed run settles its ordinary initiating message, but
				// cannot settle queued input that no persisted step causally saw.
				if (!event.preservePending) {
					pendingInputs = pendingInputs.filter((input) => input.queueId !== null);
				}
				// Keep the crash-loop count while queued recovery work survives this
				// terminal event; otherwise every pre-step failure would reset the
				// breaker and retry forever on each process restart.
				if (pendingInputs.length === 0) autoResumeAttempts = 0;
				break;
		}
	}

	const sanitized = sanitizeConversation(messages);
	const lastOpen = [...openRuns.keys()].pop() ?? null;
	const pendingQueueInputs = pendingInputs.filter(
		(input): input is { queueId: string; message: ModelMessage } => input.queueId !== null,
	);

	return {
		messages: sanitized.messages,
		contextTokens: totals.contextTokens,
		totals,
		interruptedRunId: lastOpen,
		pendingQueueIds: pendingQueueInputs.map((input) => input.queueId),
		pendingInputMessages: pendingInputs.map((input) => input.message),
		pendingMessages: pendingInputs.length,
		autoResumeAttempts,
		repaired: sanitized.removed,
	};
}
