import { Buffer } from "node:buffer";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { decodeEvent, encodeEvent } from "./serialize.js";
import type { StorageProvider, StoredEvent } from "./types.js";

/** In-memory storage — tests, throwaway one-shots. Snapshots values on both sides. */
export function memoryStorage(): StorageProvider {
	const sessions = new Map<string, string[]>();
	return {
		append(sessionId, event) {
			const encoded = encodeEvent(event);
			const events = sessions.get(sessionId);
			if (events) events.push(encoded);
			else sessions.set(sessionId, [encoded]);
		},
		load(sessionId) {
			return (sessions.get(sessionId) ?? []).map(decodeEvent);
		},
		listSessions() {
			return [...sessions.keys()];
		},
	};
}

/**
 * Serialize every append per session for custom providers whose own writes may
 * commit out of invocation order. Loads wait for the current append tail so a
 * replay cannot observe a half-drained runtime write batch.
 */
export function serializedStorage(provider: StorageProvider): StorageProvider {
	const appendChains = new Map<string, Promise<void>>();
	return {
		append(sessionId, event) {
			const previous = appendChains.get(sessionId) ?? Promise.resolve();
			const write = previous.then(() => provider.append(sessionId, event));
			const tail = write.then(
				() => undefined,
				() => undefined,
			);
			appendChains.set(sessionId, tail);
			void tail.then(() => {
				if (appendChains.get(sessionId) === tail) appendChains.delete(sessionId);
			});
			return write;
		},
		async load(sessionId) {
			await appendChains.get(sessionId);
			return provider.load(sessionId);
		},
		async listSessions() {
			await Promise.all(appendChains.values());
			return (await provider.listSessions?.()) ?? [];
		},
	};
}

// Legacy v0.5.0 name codec. Retained only so existing ledgers remain readable;
// its variable-width UTF-16 escapes are not injective for arbitrary Unicode.
function legacyFileNameFor(sessionId: string): string {
	return `${sessionId.replace(/[^a-zA-Z0-9._-]/g, (c) => `%${c.charCodeAt(0).toString(16)}`)}.jsonl`;
}

// `~` could never appear raw in a legacy name (legacy escaped it as `%7e`),
// making this prefix unambiguous. base64url over UTF-8 is bijective and path-safe.
function fileNameFor(sessionId: string): string {
	return `~${Buffer.from(sessionId, "utf8").toString("base64url")}.jsonl`;
}

function fileForSession(dir: string, sessionId: string): string {
	const current = path.join(dir, fileNameFor(sessionId));
	if (fs.existsSync(current)) return current;
	const legacy = path.join(dir, legacyFileNameFor(sessionId));
	return fs.existsSync(legacy) ? legacy : current;
}

function sessionIdForFile(name: string): string | null {
	const stem = name.slice(0, -".jsonl".length);
	if (!stem.startsWith("~")) {
		return stem.replace(/%([0-9a-f]{2})/g, (_, hex) =>
			String.fromCharCode(Number.parseInt(hex, 16)),
		);
	}
	const encoded = stem.slice(1);
	const decoded = Buffer.from(encoded, "base64url").toString("utf8");
	return Buffer.from(decoded, "utf8").toString("base64url") === encoded ? decoded : null;
}

/**
 * Default storage: one append-only JSONL file per session under `dir`.
 * Appends are serialized per session so concurrent writers can't interleave
 * half-written lines; loads tolerate a truncated final line (torn write on
 * hard kill) by dropping it.
 */
export function fileStorage(dir = "./.agentic"): StorageProvider {
	fs.mkdirSync(dir, { recursive: true });
	const appendChains = new Map<string, Promise<void>>();

	return {
		append(sessionId, event) {
			const file = fileForSession(dir, sessionId);
			const encoded = encodeEvent(event);
			const prev = appendChains.get(sessionId) ?? Promise.resolve();
			const next = prev.then(() => fsp.appendFile(file, `${encoded}\n`, "utf8"));
			// Keep the chain alive after a failed write so later appends still run.
			const tail = next.catch(() => {});
			appendChains.set(sessionId, tail);
			void tail.then(() => {
				if (appendChains.get(sessionId) === tail) appendChains.delete(sessionId);
			});
			return next;
		},
		async load(sessionId) {
			const file = fileForSession(dir, sessionId);
			let text: string;
			try {
				text = await fsp.readFile(file, "utf8");
			} catch (err) {
				if ((err as { code?: string }).code === "ENOENT") return [];
				throw err;
			}
			const events: StoredEvent[] = [];
			const lines = text.split("\n");
			for (const [index, line] of lines.entries()) {
				if (!line.trim()) continue;
				try {
					events.push(decodeEvent(line));
				} catch (error) {
					// append() terminates every complete record with a newline. Only
					// an invalid, non-newline-terminated final record can be a torn
					// write; corruption anywhere else must be visible to the caller.
					if (index === lines.length - 1 && !text.endsWith("\n")) continue;
					throw error;
				}
			}
			return events;
		},
		async listSessions() {
			const names = await fsp.readdir(dir);
			return [
				...new Set(
					names
						.filter((name) => name.endsWith(".jsonl"))
						.map(sessionIdForFile)
						.filter((id): id is string => id !== null),
				),
			];
		},
	};
}
