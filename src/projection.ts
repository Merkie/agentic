import type { ModelMessage } from "ai";
import type { SerializedError, SessionMessage, StoredEvent } from "./types.js";

export type RunProjectionStatus = "running" | "completed" | "cancelled" | "failed";

/** A durable user input associated with a projected response segment. */
export interface ProjectedInput extends SessionMessage {
	/** Position in the supplied ledger. */
	eventIndex: number;
	meta?: Record<string, unknown>;
	queued: boolean;
	queueId: string | null;
}

/**
 * One causal response segment. A live run can contain several segments when
 * new user input is accepted between model steps.
 */
export interface ProjectedRunSegment {
	inputs: ProjectedInput[];
	/** New assistant/tool messages produced for these inputs, in model order. */
	messages: SessionMessage[];
	/** Persist time of the latest response step, or the latest input. */
	at: string;
	finishReason: string | null;
	status: RunProjectionStatus;
}

export interface ProjectedRun {
	runId: string;
	status: RunProjectionStatus;
	startedAt: string;
	endedAt: string | null;
	error?: SerializedError;
	segments: ProjectedRunSegment[];
	/** Inputs that this terminal run did not consume and a later run must answer. */
	pendingInputs: ProjectedInput[];
}

interface IndexedInput extends ProjectedInput {
	settled: boolean;
}

function queueIdOf(event: Extract<StoredEvent, { type: "user-message" }>): string | null {
	const value = event.meta?.queueId;
	return event.meta?.queued === true && typeof value === "string" ? value : null;
}

function inputOf(
	event: Extract<StoredEvent, { type: "user-message" }>,
	eventIndex: number,
): IndexedInput {
	const queueId = queueIdOf(event);
	return {
		id: event.id,
		at: event.at,
		message: event.message,
		eventIndex,
		meta: event.meta,
		queued: queueId !== null,
		queueId,
		settled: false,
	};
}

/**
 * Reconstruct the user-input/assistant-response boundaries for one durable run.
 *
 * This is the package-level API storage adapters should use for a UI/chat
 * projection. It understands queued sends, step acknowledgement, interrupted
 * runs, and causal ordering. Adapters remain responsible only for choosing
 * which model messages are visible and writing the returned projection to
 * their database.
 */
export function projectRun(events: StoredEvent[], runId: string): ProjectedRun | null {
	const startIndex = events.findIndex(
		(event) => event.type === "run-start" && event.runId === runId,
	);
	if (startIndex < 0) return null;

	const start = events[startIndex] as Extract<StoredEvent, { type: "run-start" }>;
	let endIndex = -1;
	for (let index = startIndex + 1; index < events.length; index += 1) {
		const event = events[index];
		if (event?.type === "run-end" && event.runId === runId) endIndex = index;
	}
	const terminal =
		endIndex >= 0 ? (events[endIndex] as Extract<StoredEvent, { type: "run-end" }>) : null;
	const limit = endIndex >= 0 ? endIndex : events.length;

	let previousEndIndex = -1;
	for (let index = startIndex - 1; index >= 0; index -= 1) {
		if (events[index]?.type === "run-end") {
			previousEndIndex = index;
			break;
		}
	}

	const allInputs: IndexedInput[] = [];
	for (let index = 0; index < limit; index += 1) {
		const event = events[index];
		if (event?.type === "user-message") allInputs.push(inputOf(event, index));
	}
	const byId = new Map(allInputs.map((input) => [input.id, input] as const));
	const relevantInputIndexes = new Set(
		allInputs
			.filter((input) => input.eventIndex > previousEndIndex)
			.map((input) => input.eventIndex),
	);
	// A pending queued input may have been appended before the previous run's
	// terminal marker and then consumed by this successor run. Step membership
	// is authoritative, so pull referenced inputs across that run boundary.
	for (let index = startIndex + 1; index < limit; index += 1) {
		const event = events[index];
		if (event?.type !== "step" || event.runId !== runId) continue;
		for (const id of event.inputMessageIds) {
			const input = byId.get(id);
			if (input) relevantInputIndexes.add(input.eventIndex);
		}
	}
	const inputs = allInputs.filter((input) => relevantInputIndexes.has(input.eventIndex));

	type MutableSegment = Omit<ProjectedRunSegment, "status">;
	const segments: MutableSegment[] = [];
	let current: MutableSegment | null = null;
	let lastStepIndex = startIndex;

	const beginOrExtend = (included: IndexedInput[]): MutableSegment => {
		const fresh = included.filter(
			(input) => !segments.some((segment) => segment.inputs.includes(input)),
		);
		if (!current || (fresh.length > 0 && current.messages.length > 0)) {
			current = {
				inputs: [],
				messages: [],
				at: fresh.at(-1)?.at ?? start.at,
				finishReason: null,
			};
			segments.push(current);
		}
		for (const input of fresh) current.inputs.push(input);
		if (fresh.length > 0) current.at = fresh.at(-1)?.at ?? current.at;
		return current;
	};

	for (let index = startIndex + 1; index < limit; index += 1) {
		const event = events[index];
		if (event?.type !== "step" || event.runId !== runId) continue;
		lastStepIndex = index;

		let included: IndexedInput[] = [];
		if (event.acknowledgesInput) {
			included = event.inputMessageIds.flatMap((id) => {
				const input = byId.get(id);
				return input && input.eventIndex < index ? [input] : [];
			});
			for (const input of included) input.settled = true;
		} else if (!current) {
			// An error step cannot claim a queued message that arrived in flight.
			// It can still carry useful partial output for the run's initial input.
			included = inputs.filter((input) => input.eventIndex < startIndex);
		}

		const segment = beginOrExtend(included);
		segment.messages.push(
			...event.messages.map((stored) => ({
				id: stored.id,
				at: stored.at ?? event.at,
				message: stored.message,
			})),
		);
		segment.at = event.at;
		segment.finishReason = event.finishReason;
	}

	if (segments.length === 0) {
		// The process/provider may fail before a first step. Inputs already in the
		// run-start snapshot still belong before that terminal state.
		const initial = inputs.filter((input) => input.eventIndex < startIndex);
		if (initial.length > 0 || terminal) beginOrExtend(initial);
	}

	const pendingInputs = inputs.filter((input) => {
		if (input.settled) return false;
		if (!terminal) return true;
		if (input.queued) return true;
		return terminal.preservePending === true && input.eventIndex > lastStepIndex;
	});

	const status: RunProjectionStatus = terminal?.status ?? "running";
	return {
		runId,
		status,
		startedAt: start.at,
		endedAt: terminal?.at ?? null,
		...(terminal?.error ? { error: terminal.error } : {}),
		segments: segments.map((segment, index) => ({
			...segment,
			status: index === segments.length - 1 ? status : "completed",
		})),
		pendingInputs,
	};
}

// ── full-session transcript ─────────────────────────────────────────────

/**
 * Canonical text of a message: string content as-is; array content's text
 * parts joined. Non-text parts (images, tool calls, …) contribute nothing.
 */
export function textOf(message: ModelMessage): string {
	if (typeof message.content === "string") return message.content;
	if (!Array.isArray(message.content)) return "";
	return (message.content as Array<{ type?: string; text?: string }>)
		.filter((part) => part.type === "text")
		.map((part) => part.text ?? "")
		.join("");
}

export type ResponseStatus = "streaming" | "completed" | "cancelled" | "failed" | "interrupted";

export interface TranscriptUserItem {
	kind: "user";
	/** The durable ledger user-message id. */
	id: string;
	/** The run whose response consumed this input; null while queued/unclaimed. */
	runId: string | null;
	at: string;
	/** Canonical text extraction — {@link textOf} over the message. */
	text: string;
	/** Full content, for rich rendering. */
	message: ModelMessage;
	/** True while pending — no run has consumed the message yet. */
	queued: boolean;
	/** App-supplied SendOptions.meta, with the framework's queue markers stripped. */
	meta?: Record<string, unknown>;
}

export interface TranscriptResponseItem {
	kind: "response";
	/**
	 * `${runId}/${segmentIndex}` — stable from the first streaming projection
	 * to the terminal one (segments only append, so indexes never shift). Key
	 * UI rows on it; treat it as opaque.
	 */
	id: string;
	runId: string;
	/** Last update: the segment's latest step time; run start while no step yet. */
	at: string;
	/** Canonical assistant text of the whole segment. */
	text: string;
	/** The segment's stored messages (assistant + tool), for rich rendering. */
	messages: SessionMessage[];
	status: ResponseStatus;
	/** The run's failure, when status is "failed". */
	error?: SerializedError;
}

export type TranscriptItem = TranscriptUserItem | TranscriptResponseItem;

export interface SessionTranscript {
	/** Array order IS the display order. Never re-sort by timestamp. */
	items: TranscriptItem[];
	status: "idle" | "streaming" | "interrupted";
	/** The run still open in the ledger, if any. */
	activeRunId: string | null;
}

function segmentText(messages: SessionMessage[]): string {
	return messages
		.filter((entry) => entry.message.role === "assistant")
		.map((entry) => textOf(entry.message))
		.filter((text) => text.trim().length > 0)
		.join("\n\n");
}

// The framework's queue markers surface as the item's `queued`/`runId` state;
// everything else in meta is the app's own tag, passed through untouched.
function appMeta(
	event: Extract<StoredEvent, { type: "user-message" }>,
): Record<string, unknown> | undefined {
	if (!event.meta) return undefined;
	const meta = { ...event.meta };
	if (queueIdOf(event) !== null) {
		delete meta.queued;
		delete meta.queueId;
	}
	return Object.keys(meta).length > 0 ? meta : undefined;
}

/**
 * Project a session's full event ledger into a renderable transcript: user
 * turns and response segments in causal display order, with per-item
 * lifecycle status. This is THE read API for chat UIs — render the items
 * as-is; never parse raw events, enumerate runs, filter pokes, or re-sort by
 * timestamp.
 *
 * `live` says whether the calling process currently has a run executing for
 * this session (`Session.transcript()` wires it automatically): an open
 * run projects as "streaming" when live and "interrupted" when not — the
 * streaming-now vs crashed-earlier distinction the ledger alone cannot make.
 * `internal` additionally includes framework-internal user messages (pokes,
 * identifiable by `meta.poke`) for debug views. Compaction never appears:
 * summaries are model-view artifacts, not conversation.
 */
export function projectSession(
	events: StoredEvent[],
	opts: { live?: boolean; internal?: boolean } = {},
): SessionTranscript {
	const live = opts.live === true;

	interface UserEntry {
		event: Extract<StoredEvent, { type: "user-message" }>;
		index: number;
		claim: { runId: string; segmentIndex: number } | null;
		emitted: boolean;
	}
	const users: UserEntry[] = [];
	const usersById = new Map<string, UserEntry>();
	const runs: { startIndex: number; projected: ProjectedRun }[] = [];
	const openRuns = new Set<string>();

	for (let index = 0; index < events.length; index += 1) {
		const event = events[index];
		if (event?.type === "user-message") {
			const entry: UserEntry = { event, index, claim: null, emitted: false };
			users.push(entry);
			usersById.set(event.id, entry);
		} else if (event?.type === "run-start") {
			const projected = projectRun(events, event.runId);
			if (projected) runs.push({ startIndex: index, projected });
			openRuns.add(event.runId);
		} else if (event?.type === "run-end") {
			openRuns.delete(event.runId);
		}
	}

	// An input's consumer is the segment that carries it — for a queued send
	// the step that acknowledged it; for an initiating input, the run that
	// settled it. A failed run's pending input re-projects as a successor
	// segment's input; first claim wins, so it appears exactly once.
	for (const run of runs) {
		run.projected.segments.forEach((segment, segmentIndex) => {
			for (const input of segment.inputs) {
				const entry = usersById.get(input.id);
				if (entry && entry.claim === null) {
					entry.claim = { runId: run.projected.runId, segmentIndex };
				}
			}
		});
	}

	const items: TranscriptItem[] = [];
	const emitUsers = (include: (entry: UserEntry) => boolean) => {
		for (const entry of users) {
			if (entry.emitted || !include(entry)) continue;
			entry.emitted = true;
			if (entry.event.meta?.poke !== undefined && opts.internal !== true) continue;
			const meta = appMeta(entry.event);
			items.push({
				kind: "user",
				id: entry.event.id,
				runId: entry.claim?.runId ?? null,
				at: entry.event.at,
				text: textOf(entry.event.message),
				message: entry.event.message,
				queued: entry.claim === null,
				...(meta ? { meta } : {}),
			});
		}
	};

	for (const { startIndex, projected } of runs) {
		// Before a segment's response: its own inputs, plus any message that
		// arrived before this run started and is not answered by it (still
		// queued, or claimed by a later run) — all in arrival order.
		const emitSegmentUsers = (segmentIndex: number) =>
			emitUsers(
				(entry) =>
					(entry.claim !== null &&
						entry.claim.runId === projected.runId &&
						entry.claim.segmentIndex === segmentIndex) ||
					(entry.index < startIndex &&
						(entry.claim === null || entry.claim.runId !== projected.runId)),
			);
		if (projected.segments.length === 0) {
			// An open run with no step and no initiating input (queued-only
			// resume): synthesize the in-flight response so a UI has a row to
			// key on. The first step lands as segment 0, keeping the id.
			emitSegmentUsers(0);
			items.push({
				kind: "response",
				id: `${projected.runId}/0`,
				runId: projected.runId,
				at: projected.startedAt,
				text: "",
				messages: [],
				status: live ? "streaming" : "interrupted",
			});
			continue;
		}
		projected.segments.forEach((segment, segmentIndex) => {
			emitSegmentUsers(segmentIndex);
			const status: ResponseStatus =
				segment.status === "running" ? (live ? "streaming" : "interrupted") : segment.status;
			// Every step stamps finishReason, so null + no messages means no
			// step has landed for this segment yet.
			const noStepYet = segment.finishReason === null && segment.messages.length === 0;
			items.push({
				kind: "response",
				id: `${projected.runId}/${segmentIndex}`,
				runId: projected.runId,
				at: noStepYet ? projected.startedAt : segment.at,
				text: segmentText(segment.messages),
				messages: segment.messages,
				status,
				...(status === "failed" && projected.error ? { error: projected.error } : {}),
			});
		});
	}
	emitUsers(() => true);

	const activeRunId = [...openRuns].pop() ?? null;
	return {
		items,
		status: activeRunId === null ? "idle" : live ? "streaming" : "interrupted",
		activeRunId,
	};
}
