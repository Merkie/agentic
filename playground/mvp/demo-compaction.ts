// Compaction demo: pretend the model's context window is tiny so real
// provider-reported token counts cross the threshold fast, then verify the
// conversation compacts silently and the chat keeps its memory.
import "dotenv/config";
import { createAgentic, memoryStorage, setContextWindow } from "../../src/index.js";

const MODEL = "deepseek/deepseek-v4-flash";
// Pretend the window is 4k tokens; limit 0.3 → compact when context ≥ 1200.
setContextWindow(MODEL, 4_000);

const agentic = createAgentic({
	storage: memoryStorage(),
	onEvent: (e) => {
		if (e.type === "step")
			console.log(`  [step] context=${(e.usage.inputTokens ?? 0) + (e.usage.outputTokens ?? 0)}t`);
		else if (e.type === "compaction")
			console.log(`  [COMPACTION] before=${e.beforeTokens}t summary=${e.summaryChars} chars`);
		else if (e.type === "run-end") console.log(`  [run-end] ${e.status} $${e.totals.cost.toFixed(6)}`);
	},
});

const session = agentic.session("compaction-demo", {
	model: MODEL,
	system: "You are a helpful assistant. Answer in 2-3 sentences.",
	compaction: { limit: 0.3, keepRecent: 2 },
});

console.log("turn 1 (establish a fact to remember)");
await session.send("My dog's name is Biscuit and my favorite number is 42. Tell me a fun fact about dogs.");
console.log("turn 2 (bulk up the context)");
await session.send("Give me a detailed 400-word explanation of how sourdough starter works.");
console.log("turn 3 (more bulk)");
await session.send("Now a detailed 400-word history of the printing press.");
console.log("turn 4 (cross the threshold so compaction fires)");
await session.send("And a detailed 300-word explanation of photosynthesis.");
console.log("turn 5 (memory check AFTER compaction — summary is all the model has)");
const result = await session.send("What is my dog's name and my favorite number?");

console.log(`\nFINAL ANSWER: ${result.text}`);
const messages = await session.messages();
console.log(`replayed message count after compaction: ${messages.length}`);
const first = messages[0];
const preview = typeof first.content === "string" ? first.content : JSON.stringify(first.content);
console.log(`first replayed message starts with: ${preview.slice(0, 120)}...`);
const ok = /biscuit/i.test(result.text) && /42/.test(result.text);
console.log(ok ? "MEMORY SURVIVED COMPACTION ✓" : "MEMORY LOST ✗");
process.exit(ok ? 0 : 1);
