import { Buffer } from "node:buffer";
import type { StoredEvent } from "./types.js";

const BINARY_MARKER = "$agenticBinary";
const ESCAPE_MARKER = "$agenticEscaped";

function binaryBytes(value: unknown): Uint8Array | null {
	if (value instanceof Uint8Array) return value;
	if (value instanceof ArrayBuffer) return new Uint8Array(value);
	return null;
}

function ownsReservedMarker(value: object): boolean {
	return Object.hasOwn(value, BINARY_MARKER) || Object.hasOwn(value, ESCAPE_MARKER);
}

function eventReplacer(): (this: unknown, key: string, value: unknown) => unknown {
	const escapeEnvelopes = new WeakSet<object>();
	return function replaceBinary(this: unknown, key: string, value: unknown): unknown {
		// The payload of an envelope is deliberately traversed as ordinary JSON;
		// do not recursively escape the same marker-owning object again.
		if (
			this !== null &&
			typeof this === "object" &&
			escapeEnvelopes.has(this) &&
			key === ESCAPE_MARKER
		) {
			return value;
		}

		// Buffer has a toJSON() method, which runs before a JSON replacer. Read the
		// original property from the holder so Buffers are still seen as Uint8Array.
		const original = this !== null && typeof this === "object" ? Reflect.get(this, key) : value;
		const bytes = binaryBytes(original);
		if (bytes) return { [BINARY_MARKER]: Buffer.from(bytes).toString("base64") };
		if (
			original !== null &&
			typeof original === "object" &&
			!Array.isArray(original) &&
			ownsReservedMarker(original)
		) {
			const envelope = { [ESCAPE_MARKER]: original };
			escapeEnvelopes.add(envelope);
			return envelope;
		}
		return value;
	};
}

function reviveBinary(this: unknown, key: string, value: unknown): unknown {
	if (value === null || typeof value !== "object" || Array.isArray(value)) return value;
	const record = value as Record<string, unknown>;
	// This object is the payload of an escape envelope. Its children have been
	// revived, but the object itself must not be mistaken for a binary envelope.
	if (
		key === ESCAPE_MARKER &&
		this !== null &&
		typeof this === "object" &&
		Object.keys(this).length === 1 &&
		Object.hasOwn(this, ESCAPE_MARKER)
	) {
		return value;
	}
	if (Object.keys(record).length === 1 && Object.hasOwn(record, ESCAPE_MARKER)) {
		return record[ESCAPE_MARKER];
	}
	if (Object.keys(record).length !== 1 || typeof record[BINARY_MARKER] !== "string") {
		return value;
	}

	const encoded = record[BINARY_MARKER];
	const decoded = Buffer.from(encoded, "base64");
	// Only revive the canonical representation produced by encodeEvent. This
	// avoids treating most user objects that happen to use the marker as bytes.
	if (decoded.toString("base64") !== encoded) return value;
	return new Uint8Array(decoded);
}

const SCHEMA_VERSION = 1;

// The wire difference between v1 and what predates the version field is
// identity: v0.7 (which wrote no `v`) already produced every id v1 requires;
// pre-v0.7 ledgers did not. Only identity is checked — the events themselves
// are trusted, this is a version probe, not a schema validator.
function isV1Shape(event: StoredEvent): boolean {
	switch (event.type) {
		case "user-message":
			return typeof event.id === "string";
		case "step":
			return (
				Array.isArray(event.inputQueueIds) &&
				typeof event.acknowledgesInput === "boolean" &&
				Array.isArray(event.messages) &&
				event.messages.every((entry) => typeof entry?.id === "string")
			);
		case "compaction":
			return (
				!("pendingInputs" in event) &&
				Array.isArray(event.messages) &&
				event.messages.every((entry) => typeof entry?.id === "string")
			);
		default:
			return true;
	}
}

/**
 * Serialize a ledger event without losing Uint8Array, Buffer, or ArrayBuffer
 * values carried by multimodal message parts. Adapter authors should use this
 * rather than plain JSON.stringify when persisting StoredEvent values.
 * Stamps the event schema version (`v`).
 */
export function encodeEvent(event: StoredEvent): string {
	const encoded = JSON.stringify({ ...event, v: SCHEMA_VERSION }, eventReplacer());
	if (encoded === undefined) throw new TypeError("StoredEvent could not be serialized");
	return encoded;
}

/**
 * Decode a ledger event serialized by {@link encodeEvent}. The single schema
 * normalization point: every reader (replay, projection, the run loop) may
 * assume a valid v1 event after this returns. A future v2 becomes a one-time
 * migration here, never shape-sniffing in readers.
 */
export function decodeEvent(json: string): StoredEvent {
	const event = JSON.parse(json, reviveBinary) as StoredEvent;
	if (event.v === undefined) {
		// v0.7 predates the version field but already wrote v1 shapes — stamp
		// them. Anything else unversioned predates message identity entirely.
		if (!isV1Shape(event)) {
			throw new Error(
				`Unsupported ledger event ("${event.type}" without message identity): this session's data was written before agentic v0.7. Pre-v0.7 session data is unsupported.`,
			);
		}
		event.v = SCHEMA_VERSION;
		// v0.7's cancellation step omitted an empty input membership.
		if (event.type === "step" && event.inputMessageIds === undefined) event.inputMessageIds = [];
	} else if (event.v !== SCHEMA_VERSION) {
		throw new Error(
			`Unknown ledger event schema version ${event.v} (this agentic reads v${SCHEMA_VERSION}): the session was written by a newer agentic release.`,
		);
	}
	return event;
}
