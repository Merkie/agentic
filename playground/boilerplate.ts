// Boilerplate — the smallest useful harness setup. Copy this to start a
// project: one durable session with a tool, default console logging, crash
// recovery on boot. Needs OPENROUTER_API_KEY in .env.
//
//   npx tsx playground/boilerplate.ts
import "dotenv/config";
import { tool } from "ai";
import { z } from "zod";
import { createAgentic, fileStorage } from "../src/index.js";

const agentic = createAgentic({
	storage: fileStorage("./.agentic"), // JSONL ledgers, one file per session
	logs: true, // one colored console line per run/step/retry/compaction event
});

const chat = agentic.session("chat:demo-user", {
	model: "xiaomi/mimo-v2.5",
	system: "You are a helpful assistant. Keep answers short.",
	tools: {
		get_time: tool({
			description: "The current date and time.",
			inputSchema: z.object({}),
			execute: async () => new Date().toString(),
		}),
	},
	compaction: { limit: 0.3 }, // summarize at 30% of the context window
});

// After a crash or deploy, pick up any work a previous process left behind
// (interrupted runs, queued messages that never got an answer).
for (const id of await agentic.interruptedSessions()) {
	await agentic.session(id, { model: "xiaomi/mimo-v2.5" }).resume();
}

const reply = await chat.send("What time is it right now?", {
	onPart: (part) => {
		if (part.type === "text-delta") process.stdout.write(part.text);
		if (part.type === "text-end") process.stdout.write("\n");
	},
});

console.log(`\n${reply.status} — $${reply.totals.cost.toFixed(6)} total`);
