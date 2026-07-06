// Message queueing demo (opencode-style: THE QUEUE IS STORAGE).
//
// Act 1 — live queueing: while the agent is mid-run on a slow multi-step
// task, a second send() lands. The message is appended to the ledger
// immediately (durable), the run folds it in at the next step boundary, and
// both send() calls resolve with the same merged RunResult. One run-start,
// one run-end.
//
// Act 2 — durability: a user message that got queued but whose process died
// before any run touched it (simulated by appending straight to storage) is
// found by interruptedSessions() and answered by resume().
import "dotenv/config";
import { rmSync } from "node:fs";
import { tool } from "ai";
import { z } from "zod";
import { createAgentic, fileStorage } from "../../src/index.js";

const DIR = "./playground/mvp/.queue-demo";
rmSync(DIR, { recursive: true, force: true });

const agentic = createAgentic({
	storage: fileStorage(DIR),
	logs: true,
});

const agent = {
	model: "deepseek/deepseek-v4-flash",
	system:
		"You are a research assistant. Use the slow_lookup tool for every fact lookup. Answer every question the user has asked, including ones added while you were working.",
	tools: {
		slow_lookup: tool({
			description: "Look up a fact about a planet (slow external database).",
			inputSchema: z.object({ planet: z.string() }),
			execute: async ({ planet }: { planet: string }) => {
				await new Promise((r) => setTimeout(r, 2500));
				const moons: Record<string, number> = { mercury: 0, venus: 0, earth: 1, mars: 2 };
				return { planet, moons: moons[planet.toLowerCase()] ?? "unknown" };
			},
		}),
	},
};

// ── Act 1: queue into a live run ─────────────────────────────────────────
console.log("── Act 1: send #2 lands while the run is mid-task ──");
const chat = agentic.session("queue-demo", agent);

const first = chat.send(
	"Look up the moon count for Mercury, Venus, Earth, and Mars — one slow_lookup call per planet — then report all four.",
);
// let the run get properly underway, then interject
await new Promise((r) => setTimeout(r, 4000));
console.log('  >> interjecting: "Also — what is 7*6?"');
const second = chat.send("Also — quick side question: what is 7 * 6?");

const [r1, r2] = await Promise.all([first, second]);
console.log(`\nmerged run status: ${r1.status}`);
console.log(`same RunResult for both sends: ${r1 === r2}`);
console.log(`final text:\n---\n${r1.text}\n---`);

const events = await agentic.storage.load("queue-demo");
const counts = events.reduce<Record<string, number>>((acc, e) => {
	acc[e.type] = (acc[e.type] ?? 0) + 1;
	return acc;
}, {});
const queuedInLedger = events.some((e) => e.type === "user-message" && e.meta?.queued === true);
console.log(
	`ledger: ${JSON.stringify(counts)} queued-flagged message persisted: ${queuedInLedger}`,
);
if (counts["run-start"] !== 1 || counts["run-end"] !== 1) {
	throw new Error("expected the queued message to merge into ONE run");
}
if (!/42/.test(r1.text)) console.log("  (note: 7*6 answer not in final text — check transcript)");

// ── Act 2: a queued message orphaned by a crash is resumable ─────────────
console.log("\n── Act 2: orphaned queued message → interruptedSessions → resume ──");
await agentic.storage.append("orphan-demo", {
	type: "user-message",
	at: new Date().toISOString(),
	message: { role: "user", content: "How many moons does Mars have? Use slow_lookup." },
	meta: { queued: true, queueId: "crashed-before-run" },
});
const needsResume = await agentic.interruptedSessions();
console.log(`interruptedSessions(): ${JSON.stringify(needsResume)}`);
if (!needsResume.includes("orphan-demo")) throw new Error("orphaned message not detected");

const resumed = await agentic.session("orphan-demo", agent).resume();
console.log(`resume(): ${resumed?.status}`);
console.log(`answer: ${resumed?.text}`);
if (resumed?.status !== "completed") throw new Error("orphaned message was not answered");

console.log("\nPASS: queued messages merge into live runs and survive as ledger events");
