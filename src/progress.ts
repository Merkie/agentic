import type { TextStreamPart } from "ai";
import type { StreamContext } from "./projection.js";

/**
 * The canonical live-activity vocabulary, emitted by the harness itself:
 * small, JSON-serializable events an app forwards over SSE/WebSocket
 * verbatim instead of switching over raw AI SDK parts on both sides of the
 * wire. Delivered to `SendOptions.onProgress` / `TaskOptions.onProgress`
 * (the initiating caller) and to `session.attach()` listeners (every run
 * this process executes for the session).
 *
 * `tool-start.input` is the parsed tool arguments — JSON-serializable by
 * construction. Apps that want a leaner wire strip it before forwarding.
 * `retry` and `run-end` are live-only lifecycle markers; the durable record
 * of terminal state remains `RunResult`/the transcript.
 */
export type ProgressEvent =
	/**
	 * Start of every model pass. Deltas that follow belong to `responseId` and
	 * their offsets restart at 0. Every model request announces itself: fresh
	 * passes, retry passes, the later steps of a tool loop, and passes
	 * continuing the same segment — the event is the pass boundary marker, so
	 * its `responseId` may repeat.
	 */
	| { type: "response-start"; runId: string; responseId: string }
	| { type: "text"; runId: string; responseId: string; delta: string; offset: number }
	| { type: "reasoning"; runId: string; responseId: string; delta: string }
	| {
			type: "tool-start";
			runId: string;
			responseId: string;
			toolCallId: string;
			toolName: string;
			input: unknown;
	  }
	| { type: "tool-end"; runId: string; responseId: string; toolCallId: string; toolName: string }
	/** Live-only: a transient failure is being retried (renderable as "retrying…"). */
	| { type: "retry"; runId: string; attempt: number }
	| { type: "run-end"; runId: string; status: "completed" | "cancelled" | "failed" };

/**
 * Package-internal: map one live stream part to its activity event, or null
 * for parts with no activity meaning (start-step, finish, usage bookkeeping,
 * …). The run loop applies it at the part-delivery site; apps never call it —
 * they receive the results through onProgress/attach.
 *
 * `tool-start` maps from `tool-call`, not `tool-input-start`: it is the one
 * part every provider emits exactly once per invocation (input-streaming
 * parts exist only when the model streams arguments incrementally, and a
 * stateless mapper cannot dedupe the two), and it opens the execution window
 * that a non-preliminary `tool-result` closes as `tool-end`.
 */
export function partProgress(
	// biome-ignore lint/suspicious/noExplicitAny: contravariant position — parts of any ToolSet map
	part: TextStreamPart<any>,
	context: StreamContext,
): ProgressEvent | null {
	const { runId, responseId } = context;
	switch (part.type) {
		case "text-delta":
			return { type: "text", runId, responseId, delta: part.text, offset: context.offset ?? 0 };
		case "reasoning-delta":
			return { type: "reasoning", runId, responseId, delta: part.text };
		case "tool-call":
			return {
				type: "tool-start",
				runId,
				responseId,
				toolCallId: part.toolCallId,
				toolName: part.toolName,
				input: part.input,
			};
		case "tool-result":
			// A preliminary result is a replaceable preview — the tool is still
			// running; only its authoritative result ends the activity window.
			if (part.preliminary === true) return null;
			return {
				type: "tool-end",
				runId,
				responseId,
				toolCallId: part.toolCallId,
				toolName: part.toolName,
			};
		default:
			return null;
	}
}
