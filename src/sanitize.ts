// Heal a persisted ModelMessage[] before replaying it to a provider. A run
// interrupted mid tool-call (crash, sever, abort) can leave: half-formed tool
// parts with empty names/ids, duplicated tool parts, or assistant tool-calls
// with no answering tool-result — all of which providers reject with
// deterministic 400s (so the broken history would fail identically on every
// replay until repaired). Healing on load makes it self-heal: the next
// successful persist snapshots the cleaned history.
//
// Ported from halo-cmo's conversationSanitizer, which earned each rule from a
// real production failure.

interface ModelMessageLike {
	role?: string;
	content?: unknown;
	[k: string]: unknown;
}

type Part = {
	type?: string;
	toolName?: unknown;
	toolCallId?: unknown;
	[k: string]: unknown;
};

function nonEmptyString(v: unknown): v is string {
	return typeof v === "string" && v.length > 0;
}

function isMalformedPart(part: unknown): boolean {
	if (!part || typeof part !== "object") return false;
	const p = part as Part;
	if (p.type === "tool-call" && !nonEmptyString(p.toolName)) return true;
	if (p.type === "tool-result" && !nonEmptyString(p.toolCallId)) return true;
	return false;
}

function partsOf(m: ModelMessageLike | undefined): Part[] {
	return m && Array.isArray(m.content) ? (m.content as Part[]) : [];
}

function isToolResultMessage(m: ModelMessageLike | undefined): boolean {
	return m?.role === "tool" && partsOf(m).some((p) => p?.type === "tool-result");
}

/**
 * Repair a saved conversation so it is safe to replay:
 *
 *  - drops tool-call parts with no `toolName` / tool-results with no id;
 *  - dedupes tool parts by `toolCallId` within a message (keep first);
 *  - enforces tool-call/result adjacency: assistant tool-calls survive only
 *    if answered in the immediately-following tool messages, and orphan tool
 *    results are dropped;
 *  - drops messages whose content array becomes empty.
 *
 * Idempotent and a no-op on already-valid conversations.
 */
export function sanitizeConversation<T extends ModelMessageLike>(
	messages: T[],
): { messages: T[]; removed: number } {
	if (!Array.isArray(messages)) return { messages, removed: 0 };
	let removed = 0;

	// Pass 1 — per message: drop malformed parts + dedupe tool parts by id.
	const cleaned = messages.map((msg) => {
		if (!msg || !Array.isArray(msg.content)) return msg;
		const seen = new Set<string>();
		const kept = (msg.content as Part[]).filter((p) => {
			if (!p || typeof p !== "object") return true;
			if (isMalformedPart(p)) {
				removed += 1;
				return false;
			}
			if (p.type === "tool-call" || p.type === "tool-result") {
				const key = `${p.type}:${String(p.toolCallId)}`;
				if (seen.has(key)) {
					removed += 1;
					return false;
				}
				seen.add(key);
			}
			return true;
		});
		return kept.length === (msg.content as unknown[]).length
			? msg
			: ({ ...msg, content: kept } as T);
	});

	// Pass 2 — adjacency pairing.
	const out: T[] = [];
	for (let i = 0; i < cleaned.length; i += 1) {
		const msg = cleaned[i];
		const parts = partsOf(msg);
		const callIds = parts
			.filter((p) => p?.type === "tool-call")
			.map((p) => p.toolCallId as string);

		if (msg?.role === "assistant" && callIds.length > 0) {
			const followers: number[] = [];
			const provided = new Set<string>();
			let j = i + 1;
			while (j < cleaned.length && isToolResultMessage(cleaned[j])) {
				followers.push(j);
				for (const p of partsOf(cleaned[j])) {
					if (p?.type === "tool-result") provided.add(p.toolCallId as string);
				}
				j += 1;
			}
			const keep = new Set(callIds.filter((id) => provided.has(id)));

			const aKept = parts.filter((p) => {
				if (p?.type === "tool-call" && !keep.has(p.toolCallId as string)) {
					removed += 1;
					return false;
				}
				return true;
			});
			if (aKept.length > 0) {
				out.push(aKept.length === parts.length ? msg : ({ ...msg, content: aKept } as T));
			}

			for (const f of followers) {
				const fParts = partsOf(cleaned[f]);
				const fKept = fParts.filter((p) => {
					if (p?.type === "tool-result" && !keep.has(p.toolCallId as string)) {
						removed += 1;
						return false;
					}
					return true;
				});
				if (fKept.length > 0) {
					out.push(
						fKept.length === fParts.length
							? cleaned[f]
							: ({ ...cleaned[f], content: fKept } as T),
					);
				}
			}
			i = j - 1;
			continue;
		}

		if (isToolResultMessage(msg)) {
			// Orphan tool message (no preceding matching assistant) — drop results.
			const kept = parts.filter((p) => {
				if (p?.type === "tool-result") {
					removed += 1;
					return false;
				}
				return true;
			});
			if (kept.length > 0) {
				out.push(kept.length === parts.length ? msg : ({ ...msg, content: kept } as T));
			}
			continue;
		}

		if (Array.isArray(msg?.content) && (msg!.content as unknown[]).length === 0) {
			continue;
		}

		out.push(msg);
	}

	return { messages: out, removed };
}
