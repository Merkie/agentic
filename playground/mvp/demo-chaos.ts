// Resilience demo: inject provider 500s AND a mid-SSE connection sever into a
// tool-using run — the run must complete anyway via classify + backoff +
// replay-from-storage, with every finished step preserved.
import "dotenv/config";
import { z } from "zod";
import { tool } from "ai";
import { createAgentic, memoryStorage } from "../../src/index.js";

let requestCount = 0;
const chaosFetch: typeof fetch = async (input, init) => {
	requestCount++;
	if (requestCount === 2) {
		console.log(`  [chaos] request #${requestCount}: returning 500`);
		return new Response(JSON.stringify({ error: { message: "simulated provider meltdown", code: 500 } }), {
			status: 500,
			headers: { "content-type": "application/json" },
		});
	}
	const res = await fetch(input, init);
	if (requestCount === 3 && res.body) {
		console.log(`  [chaos] request #${requestCount}: will sever SSE after 500 bytes`);
		let seen = 0;
		const reader = res.body.getReader();
		const body = new ReadableStream<Uint8Array>({
			async pull(controller) {
				const { done, value } = await reader.read();
				if (done) return controller.close();
				seen += value.byteLength;
				controller.enqueue(value);
				if (seen > 500) {
					console.log("  [chaos] SEVERED");
					controller.error(new Error("ECONNRESET: simulated death mid-stream"));
					reader.cancel().catch(() => {});
				}
			},
		});
		return new Response(body, { status: res.status, headers: res.headers });
	}
	return res;
};

const agentic = createAgentic({
	storage: memoryStorage(),
	fetchTimeouts: { fetch: chaosFetch },
	onEvent: (e) => {
		if (e.type === "step")
			console.log(`  [step] ${e.finishReason} tools=${e.toolCalls.map((t) => t.toolName).join(",") || "-"}`);
		else if (e.type === "retry")
			console.log(`  [retry ${e.attempt}/${e.maxAttempts}] wait ${e.delayMs}ms — ${e.error.message?.slice(0, 80)}`);
		else if (e.type === "run-end") console.log(`  [run-end] ${e.status}`);
	},
});

const session = agentic.session("chaos-demo", {
	model: "xiaomi/mimo-v2.5",
	system: "You are a helpful assistant.",
	tools: {
		get_weather: tool({
			description: "Get current weather for a city",
			inputSchema: z.object({ city: z.string() }),
			execute: async ({ city }) => ({ city, tempC: city === "Tokyo" ? 28 : 19 }),
		}),
	},
	retry: { baseDelayMs: 1_000 },
});

const result = await session.send(
	"Get the weather in Tokyo, then Paris (one at a time), then give me a one-line comparison.",
);
console.log(`\nRESULT: ${result.status} | requests made: ${requestCount}`);
console.log(`TEXT: ${result.text}`);
console.log(`replayed messages: ${(await session.messages()).length}`);
if (result.status !== "completed") process.exit(1);
