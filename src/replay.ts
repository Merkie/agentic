import type { ModelMessage } from "ai";
import { sanitizeConversation } from "./sanitize.js";
import type { SessionMessage, StoredEvent, UsageTotals } from "./types.js";
import { addStepToTotals, emptyTotals } from "./usage.js";

// Identity rides through replay on symbol tags so sanitizeConversation's
// clone-on-repair ({ ...msg, content }) carries it automatically; the tags are
// stripped when the public SessionMessage envelopes are built.
const MESSAGE_ID = Symbol("agenticMessageId");
const MESSAGE_AT = Symbol("agenticMessageAt");

type TaggedMessage = ModelMessage & {
	[MESSAGE_ID]?: string;
	[MESSAGE_AT]?: string;
};

function tagMessage(message: ModelMessage, id: string, at: string): TaggedMessage {
	return { ...message, [MESSAGE_ID]: id, [MESSAGE_AT]: at };
}

function untagMessage(tagged: TaggedMessage): SessionMessage {
	const message = { ...tagged };
	delete message[MESSAGE_ID];
	delete message[MESSAGE_AT];
	// Every replayed message goes through tagMessage, so the fallbacks are for
	// the type system only.
	return {
		id: tagged[MESSAGE_ID] ?? "",
		at: tagged[MESSAGE_AT] ?? "",
		message: message as ModelMessage,
	};
}

interface PendingInput {
	queueId: string | null;
	/** The ledger message id. */
	messageId: string;
	/** Live reference into the working conversation, for causal reordering. */
	tagged: TaggedMessage;
}

function moveInputsToTail(messages: TaggedMessage[], inputs: PendingInput[]): void {
	const tail: TaggedMessage[] = [];
	for (const input of inputs) {
		const index = messages.indexOf(input.tagged);
		if (index >= 0) messages.splice(index, 1);
		tail.push(input.tagged);
	}
	messages.push(...tail);
}

export interface ReplayedSession {
	/** The replay-ready conversation (sanitized), each message with its ledger id and persist time. */
	messages: SessionMessage[];
	/** Latest step's context size (input+output tokens) — drives compaction. */
	contextTokens: number | null;
	/** Lifetime totals across every run and compaction in the ledger. */
	totals: UsageTotals;
	/** runId of a run that started but never ended (crash/kill), if any. */
	interruptedRunId: string | null;
	/** Queued user-message ids not yet included in a persisted model step. */
	pendingQueueIds: string[];
	/** Exact unanswered input messages, in arrival order, for causal prompt hoisting. */
	pendingInputMessages: SessionMessage[];
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
	let messages: TaggedMessage[] = [];
	let totals = emptyTotals();
	let pendingInputs: PendingInput[] = [];
	let autoResumeAttempts = 0;
	const openRuns = new Map<string, true>();
	// Snapshot of the working state for the currently-open run (there is at
	// most one), so a `discarded` run-end can rewind the run's conversational
	// contribution. Entries are cloned: compaction re-links the LIVE entries'
	// tagged refs, and a rewind must restore the refs as of run-start/arrival.
	let openRun: {
		runId: string;
		messages: TaggedMessage[];
		pendingInputs: PendingInput[];
		/** User messages appended while the run was open, in arrival order. */
		arrivals: { entry: PendingInput; poke: boolean }[];
	} | null = null;

	for (const event of events) {
		switch (event.type) {
			case "user-message": {
				const rawQueueId = event.meta?.queueId;
				const queueId =
					event.meta?.queued === true && typeof rawQueueId === "string" ? rawQueueId : null;
				const tagged = tagMessage(event.message, event.id, event.at);
				messages.push(tagged);
				pendingInputs.push({ queueId, messageId: event.id, tagged });
				if (openRun) {
					openRun.arrivals.push({
						entry: { queueId, messageId: event.id, tagged },
						poke: event.meta?.poke !== undefined,
					});
				}
				break;
			}
			case "step": {
				if (event.acknowledgesInput) {
					const included = new Set(event.inputQueueIds);
					moveInputsToTail(
						messages,
						pendingInputs.filter((input) => input.queueId === null || included.has(input.queueId)),
					);
					pendingInputs = pendingInputs.filter(
						(input) => input.queueId !== null && !included.has(input.queueId),
					);
				}
				for (const stored of event.messages) {
					messages.push(tagMessage(stored.message, stored.id, stored.at ?? event.at));
				}
				totals = addStepToTotals(totals, event.usage);
				break;
			}
			case "compaction": {
				messages = event.messages.map((stored) =>
					tagMessage(stored.message, stored.id, stored.at ?? event.at),
				);
				// Pending inputs preserved verbatim in the base must keep their exact
				// identity: re-link each entry to its rebased message by ledger id.
				const rebasedById = new Map<string, TaggedMessage>();
				for (const tagged of messages) {
					const id = tagged[MESSAGE_ID];
					if (id !== undefined) rebasedById.set(id, tagged);
				}
				for (const input of pendingInputs) {
					const rebased = rebasedById.get(input.messageId);
					if (rebased) input.tagged = rebased;
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
				openRun = {
					runId: event.runId,
					messages: [...messages],
					pendingInputs: pendingInputs.map((input) => ({ ...input })),
					arrivals: [],
				};
				break;
			case "run-resume":
				if (event.auto) autoResumeAttempts += 1;
				break;
			case "run-end":
				openRuns.delete(event.runId);
				if (event.discarded && openRun?.runId === event.runId) {
					// REWIND: the run's conversational contribution is void. Restore
					// the run-start working state, then re-append what arrived during
					// the run (in arrival order) to messages AND pending — even inputs
					// the dead run acknowledged, because a discarded run's
					// acknowledgments are void too. Mid-run compaction vanishes with
					// the rewind (the rebased base and its summary were this run's
					// contribution). Pokes are NOT restored: a poke's purpose —
					// prodding THIS run to settle — died with the run, and replaying
					// it would nag the fresh run with a stale instruction. Usage
					// totals stay (the cost was real); the live context size is
					// unknown until the fresh run's first step reports usage.
					const restored = openRun.arrivals.filter((arrival) => !arrival.poke);
					messages = [...openRun.messages, ...restored.map((arrival) => arrival.entry.tagged)];
					pendingInputs = [...openRun.pendingInputs, ...restored.map((arrival) => arrival.entry)];
					totals = { ...totals, contextTokens: null };
				} else if (!event.preservePending && !event.discarded) {
					// A cancelled/failed run settles its ordinary initiating message, but
					// cannot settle queued input that no persisted step causally saw.
					pendingInputs = pendingInputs.filter((input) => input.queueId !== null);
				}
				openRun = null;
				// Keep the crash-loop count while queued recovery work survives this
				// terminal event; otherwise every pre-step failure would reset the
				// breaker and retry forever on each process restart.
				if (pendingInputs.length === 0) autoResumeAttempts = 0;
				break;
		}
	}

	const sanitized = sanitizeConversation(messages);
	const lastOpen = [...openRuns.keys()].pop() ?? null;

	const envelopeOf = new Map<TaggedMessage, SessionMessage>();
	const finalMessages = sanitized.messages.map((tagged) => {
		const envelope = untagMessage(tagged);
		envelopeOf.set(tagged, envelope);
		return envelope;
	});
	// Pending refs almost always survive sanitization untouched (user messages
	// are never tool-shaped). If one was cloned by a repair, recover it by id;
	// a message dropped entirely still surfaces so hoisting can re-append it.
	const pendingInputMessages = pendingInputs.map((input) => {
		const direct = envelopeOf.get(input.tagged);
		if (direct) return direct;
		const byId = finalMessages.find((candidate) => candidate.id === input.messageId);
		return byId ?? untagMessage(input.tagged);
	});

	return {
		messages: finalMessages,
		contextTokens: totals.contextTokens,
		totals,
		interruptedRunId: lastOpen,
		pendingQueueIds: pendingInputs.flatMap((input) => (input.queueId ? [input.queueId] : [])),
		pendingInputMessages,
		pendingMessages: pendingInputs.length,
		autoResumeAttempts,
		repaired: sanitized.removed,
	};
}
