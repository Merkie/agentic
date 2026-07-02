// ═══════════════════════════════════════════════════════════════════════
// BEFORE: the banker workflow on the raw AI SDK.
//
// This is an honest, condensed port of what halo-cmo has to do today for
// every workflow agent (websiteAgent/videoAgent + agentResilience +
// agentLedger + conversationSanitizer + openRouterClient, ~3,000 lines
// spread over 10 files). Everything below the "domain code" marker is pure
// plumbing that gets re-written, slightly differently, for every new agent —
// and every new agent re-discovers the same bugs.
// ═══════════════════════════════════════════════════════════════════════
import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { type ModelMessage, stepCountIs, streamText, tool } from "ai";
import { z } from "zod";

// ── domain code (what the task is actually about) ────────────────────────
const MODEL = "deepseek/deepseek-v4-flash";
const ACCOUNTS: Record<string, { owner: string; balance: number }> = {
	"ACC-1001": { owner: "John Smith", balance: 4211.5 },
	"ACC-2002": { owner: "Mary Sue", balance: 91_224.01 },
};
const deliverableSchema = z.object({
	accountId: z.string(),
	balance: z.number(),
	formatted: z.string(),
});
const TASK_PROMPT = `Customer John Smith (authenticated as owner of ACC-1001) asks: "What's my checking account balance?" Look it up and submit the balance.`;

// ── plumbing layer 1: resilient fetch (header + SSE-idle timeouts) ───────
class HeaderTimeoutError extends Error {
	override name = "HeaderTimeoutError";
}
class StreamStallError extends Error {
	override name = "StreamStallError";
}
function resilientFetch(headerMs = 60_000, chunkMs = 120_000): typeof fetch {
	return (async (input: RequestInfo | URL, init?: RequestInit) => {
		const headerCtl = new AbortController();
		const headerTimer = setTimeout(
			() => headerCtl.abort(new HeaderTimeoutError("headers timed out")),
			headerMs,
		);
		const signals = [headerCtl.signal, ...(init?.signal ? [init.signal as AbortSignal] : [])];
		try {
			const res = await fetch(input, { ...init, signal: AbortSignal.any(signals) });
			clearTimeout(headerTimer);
			if (!res.body || !(res.headers.get("content-type") ?? "").includes("event-stream")) return res;
			const reader = res.body.getReader();
			const body = new ReadableStream<Uint8Array>({
				async pull(controller) {
					let timer: ReturnType<typeof setTimeout> | undefined;
					try {
						const chunk = await Promise.race([
							reader.read(),
							new Promise<never>((_, rej) => {
								timer = setTimeout(() => rej(new StreamStallError("SSE read timed out")), chunkMs);
							}),
						]);
						if (chunk.done) controller.close();
						else controller.enqueue(chunk.value);
					} catch (err) {
						reader.cancel().catch(() => {});
						controller.error(err);
					} finally {
						clearTimeout(timer);
					}
				},
			});
			return new Response(body, { status: res.status, headers: res.headers });
		} finally {
			clearTimeout(headerTimer);
		}
	}) as typeof fetch;
}

// ── plumbing layer 2: transient-vs-fatal error classifier ────────────────
const TRANSIENT_NEEDLES = ["econnreset", "socket hang up", "timeout", "timed out", "overloaded",
	"bad gateway", "service unavailable", "rate limit", "too many requests", "internal server error",
	"upstream idle", "provider returned", "no output generated", "fetch failed", "terminated"];
const FATAL_NEEDLES = ["credits", "payment required", "billing", "invalid api key", "unauthorized",
	"content policy", "param incorrect", "invalid_request"];
function isTransient(err: unknown): boolean {
	let s: string;
	try {
		s = (err instanceof Error ? `${err.name} ${err.message}` : JSON.stringify(err)).toLowerCase();
	} catch {
		s = String(err).toLowerCase();
	}
	if (FATAL_NEEDLES.some((n) => s.includes(n))) return false;
	if (/"(?:status|statuscode|code)":\s*"?(?:429|5\d\d)"?/.test(s)) return true;
	return TRANSIENT_NEEDLES.some((n) => s.includes(n));
}

// ── plumbing layer 3: JSONL step ledger + replay (survive restarts) ──────
const LEDGER = "./playground/mvp/before-after/.before-ledger.jsonl";
function ledgerAppend(row: Record<string, unknown>) {
	fs.mkdirSync(path.dirname(LEDGER), { recursive: true });
	fs.appendFileSync(LEDGER, `${JSON.stringify(row)}\n`);
}
function ledgerReplay(): ModelMessage[] {
	if (!fs.existsSync(LEDGER)) return [];
	const messages: ModelMessage[] = [];
	for (const line of fs.readFileSync(LEDGER, "utf8").split("\n")) {
		if (!line.trim()) continue;
		try {
			const row = JSON.parse(line);
			if (row.kind === "user_message") messages.push(row.message);
			if (row.kind === "step_messages") messages.push(...row.messages);
		} catch {} // torn write on crash
	}
	return sanitize(messages);
}
// ── plumbing layer 4: transcript repair (crash mid tool-call → 400s) ─────
function sanitize(messages: ModelMessage[]): ModelMessage[] {
	const out: ModelMessage[] = [];
	for (let i = 0; i < messages.length; i++) {
		const m = messages[i];
		if (m.role === "assistant" && Array.isArray(m.content)) {
			const calls = m.content.filter((p) => (p as { type?: string }).type === "tool-call");
			if (calls.length > 0) {
				const next = messages[i + 1];
				const answered = new Set(
					next?.role === "tool" && Array.isArray(next.content)
						? next.content.map((p) => (p as { toolCallId?: string }).toolCallId)
						: [],
				);
				const kept = m.content.filter(
					(p) =>
						(p as { type?: string }).type !== "tool-call" ||
						answered.has((p as { toolCallId?: string }).toolCallId),
				);
				if (kept.length > 0) out.push({ ...m, content: kept } as ModelMessage);
				continue;
			}
		}
		out.push(m);
	}
	return out;
}

// ── plumbing layer 5: usage/cost accounting (BYOK vs credits) ────────────
let totalCost = 0;
function trackCost(providerMetadata: unknown) {
	const u = (providerMetadata as { openrouter?: { usage?: { cost?: number; costDetails?: { upstreamInferenceCost?: number | null } } } })
		?.openrouter?.usage;
	totalCost += (u?.cost ?? 0) + (u?.costDetails?.upstreamInferenceCost ?? 0);
}

// ── plumbing layer 6: the retry/poke/outcome driver loop ─────────────────
const openrouter = createOpenRouter({
	apiKey: process.env.OPENROUTER_API_KEY,
	extraBody: { usage: { include: true } },
	fetch: resilientFetch(),
});

type Outcome =
	| { status: "submitted"; deliverable: z.infer<typeof deliverableSchema> }
	| { status: "cancelled"; reason: string };
let outcome: Outcome | null = null;

const tools = {
	get_account: tool({
		description: "Look up an account by id",
		inputSchema: z.object({ accountId: z.string() }),
		execute: async ({ accountId }) => ACCOUNTS[accountId] ?? { error: "not found" },
	}),
	submit_deliverable: tool({
		description: `Submit the final result. Must match: ${JSON.stringify(z.toJSONSchema(deliverableSchema))}`,
		inputSchema: z.object({ deliverable: z.unknown() }),
		execute: async ({ deliverable }) => {
			let v = deliverable;
			if (typeof v === "string") {
				try {
					v = JSON.parse(v);
				} catch {}
			}
			const check = deliverableSchema.safeParse(v);
			if (!check.success) return { accepted: false, error: check.error.message };
			outcome = { status: "submitted", deliverable: check.data };
			return { accepted: true };
		},
	}),
	cancel_task: tool({
		description: "Cancel an impossible/unauthorized task",
		inputSchema: z.object({ reason: z.string() }),
		execute: async ({ reason }) => {
			outcome = { status: "cancelled", reason };
			return { ok: true };
		},
	}),
};

async function main() {
	fs.rmSync(LEDGER, { force: true });
	ledgerAppend({ kind: "user_message", message: { role: "user", content: TASK_PROMPT } });

	let attempts = 0;
	let pokes = 0;
	while (!outcome) {
		const messages = ledgerReplay();
		let streamError: unknown;
		const result = streamText({
			model: openrouter.chat(MODEL),
			system:
				"You are a bank task worker. You MUST end by calling submit_deliverable or cancel_task.",
			messages,
			tools,
			maxRetries: 0, // we own retry policy
			stopWhen: [stepCountIs(10), () => outcome !== null],
			onError: () => {},
			onStepFinish: (step) => {
				trackCost(step.providerMetadata);
				ledgerAppend({ kind: "step_messages", messages: step.response.messages });
			},
		});
		try {
			for await (const part of result.fullStream) {
				if (part.type === "error") streamError = part.error;
			}
		} catch (err) {
			streamError ??= err;
		}

		if (streamError) {
			if (!isTransient(streamError)) throw streamError;
			attempts++;
			if (attempts > 5) throw new Error(`gave up after ${attempts} transient failures`);
			const delay = Math.min(2000 * 2 ** (attempts - 1), 60_000);
			console.log(`  transient failure, retry ${attempts} in ${delay}ms`);
			await new Promise((r) => setTimeout(r, delay));
			continue;
		}
		if (!outcome) {
			pokes++;
			if (pokes > 2) throw new Error("model never called a terminal tool");
			console.log(`  poke ${pokes}`);
			ledgerAppend({
				kind: "user_message",
				message: {
					role: "user",
					content: "You ended without calling submit_deliverable or cancel_task. Call one now.",
				},
			});
		}
	}
	console.log("OUTCOME:", JSON.stringify(outcome));
	console.log(`cost: $${totalCost.toFixed(6)}`);
}
main();
