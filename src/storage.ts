import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import type { StorageProvider, StoredEvent } from "./types.js";

/** In-memory storage — tests, throwaway one-shots. */
export function memoryStorage(): StorageProvider {
	const sessions = new Map<string, StoredEvent[]>();
	return {
		append(sessionId, event) {
			const events = sessions.get(sessionId);
			if (events) events.push(event);
			else sessions.set(sessionId, [event]);
		},
		load(sessionId) {
			return [...(sessions.get(sessionId) ?? [])];
		},
		listSessions() {
			return [...sessions.keys()];
		},
	};
}

// Session ids become file names; anything path-hostile is escaped so ids like
// "chat:user/123" can't traverse directories or collide.
function fileNameFor(sessionId: string): string {
	return `${sessionId.replace(/[^a-zA-Z0-9._-]/g, (c) => `%${c.charCodeAt(0).toString(16)}`)}.jsonl`;
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
			const file = path.join(dir, fileNameFor(sessionId));
			const prev = appendChains.get(sessionId) ?? Promise.resolve();
			const next = prev.then(() => fsp.appendFile(file, `${JSON.stringify(event)}\n`, "utf8"));
			// Keep the chain alive after a failed write so later appends still run.
			appendChains.set(
				sessionId,
				next.catch(() => {}),
			);
			return next;
		},
		async load(sessionId) {
			const file = path.join(dir, fileNameFor(sessionId));
			let text: string;
			try {
				text = await fsp.readFile(file, "utf8");
			} catch (err) {
				if ((err as { code?: string }).code === "ENOENT") return [];
				throw err;
			}
			const events: StoredEvent[] = [];
			for (const line of text.split("\n")) {
				if (!line.trim()) continue;
				try {
					events.push(JSON.parse(line) as StoredEvent);
				} catch {
					// torn final line from a hard kill — drop it; every complete
					// event before it is intact
				}
			}
			return events;
		},
		async listSessions() {
			const names = await fsp.readdir(dir);
			return names
				.filter((n) => n.endsWith(".jsonl"))
				.map((n) =>
					n
						.slice(0, -".jsonl".length)
						.replace(/%([0-9a-f]{2})/g, (_, hex) => String.fromCharCode(Number.parseInt(hex, 16))),
				);
		},
	};
}
