import type { TextStreamPart } from "ai";
import type { StreamContext } from "./projection.js";

/**
 * The canonical live-activity vocabulary: small, JSON-serializable events an
 * app can forward over SSE/WebSocket verbatim instead of switching over raw
 * AI SDK parts on both sides of the wire. Deliberately lean — no tool
 * inputs/outputs (apps that want payloads read the raw parts in
 * onPart/attach), and no terminal/lifecycle events: completion, failure, and
 * cancellation already arrive through onAccepted/RunResult/the transcript.
 */
export type ProgressEvent =
	| { type: "text"; responseId: string; runId: string; delta: string; offset: number }
	| { type: "reasoning"; responseId: string; runId: string; delta: string }
	| { type: "tool-start"; responseId: string; runId: string; toolCallId: string; toolName: string }
	| { type: "tool-end"; responseId: string; runId: string; toolCallId: string; toolName: string };

/**
 * Map one live stream part to its {@link ProgressEvent}, or null for parts
 * with no activity meaning (start-step, finish, usage bookkeeping, …). Pure
 * and stateless — apply it inside an onPart/attach listener and forward the
 * non-null results; `offset` on text events passes through from
 * {@link StreamContext} for exact reconnect dedupe against `partialText`.
 *
 * `tool-start` maps from `tool-call`, not `tool-input-start`: it is the one
 * part every provider emits exactly once per invocation (input-streaming
 * parts exist only when the model streams arguments incrementally, and a
 * stateless mapper cannot dedupe the two), and it opens the execution window
 * that a non-preliminary `tool-result` closes as `tool-end`.
 */
export function progressFromPart(
	// biome-ignore lint/suspicious/noExplicitAny: contravariant position — parts of any ToolSet map
	part: TextStreamPart<any>,
	context: StreamContext,
): ProgressEvent | null {
	const { runId, responseId } = context;
	switch (part.type) {
		case "text-delta":
			return { type: "text", responseId, runId, delta: part.text, offset: context.offset ?? 0 };
		case "reasoning-delta":
			return { type: "reasoning", responseId, runId, delta: part.text };
		case "tool-call":
			return {
				type: "tool-start",
				responseId,
				runId,
				toolCallId: part.toolCallId,
				toolName: part.toolName,
			};
		case "tool-result":
			// A preliminary result is a replaceable preview — the tool is still
			// running; only its authoritative result ends the activity window.
			if (part.preliminary === true) return null;
			return {
				type: "tool-end",
				responseId,
				runId,
				toolCallId: part.toolCallId,
				toolName: part.toolName,
			};
		default:
			return null;
	}
}
