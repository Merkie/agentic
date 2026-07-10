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

/**
 * Serialize a ledger event without losing Uint8Array, Buffer, or ArrayBuffer
 * values carried by multimodal message parts. Adapter authors should use this
 * rather than plain JSON.stringify when persisting StoredEvent values.
 */
export function encodeEvent(event: StoredEvent): string {
	const encoded = JSON.stringify(event, eventReplacer());
	if (encoded === undefined) throw new TypeError("StoredEvent could not be serialized");
	return encoded;
}

/** Decode a ledger event serialized by {@link encodeEvent}. */
export function decodeEvent(json: string): StoredEvent {
	return JSON.parse(json, reviveBinary) as StoredEvent;
}
