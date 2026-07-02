// Server-restart survival demo. Phase "start": kick off a long multi-tool run
// with file storage and hard-kill the process mid-run (SIGKILL — no cleanup).
// Phase "resume": new process finds the interrupted session in storage and
// resumes the SAME run to completion from the last persisted step.
import "dotenv/config";
import { spawn } from "node:child_process";
import { tool } from "ai";
import { z } from "zod";
import { createAgentic, fileStorage } from "../../src/index.js";

const DIR = "./playground/mvp/.restart-demo";
const phase = process.argv[2] ?? "orchestrate";

function buildAgentic() {
	return createAgentic({
		storage: fileStorage(DIR),
		onEvent: (e) => {
			if (e.type === "step")
				console.log(
					`  [step] ${e.finishReason} tools=${e.toolCalls.map((t) => t.toolName).join(",") || "-"}`,
				);
			else if (e.type === "run-end") console.log(`  [run-end] ${e.status}`);
		},
	});
}

const agentConfig = {
	model: "xiaomi/mimo-v2.5",
	system: "You are a research assistant. Look up each city one at a time.",
	tools: {
		lookup_city: tool({
			description: "Look up facts about a city (slow database)",
			inputSchema: z.object({ city: z.string() }),
			execute: async ({ city }: { city: string }) => {
				await new Promise((r) => setTimeout(r, 1500)); // slow tool → easy kill window
				return { city, population: city.length * 1_000_000, nice: true };
			},
		}),
	},
};

const PROMPT =
	"Look up these cities one at a time with lookup_city: Tokyo, Paris, Berlin, Madrid, Rome. Then summarize which is largest.";

if (phase === "start") {
	const session = buildAgentic().session("restart-demo", agentConfig);
	const result = await session.send(PROMPT);
	console.log("completed without being killed:", result.status);
} else if (phase === "resume") {
	const agentic = buildAgentic();
	const interrupted = await agentic.interruptedSessions();
	console.log("interrupted sessions found:", interrupted);
	const session = agentic.session("restart-demo", agentConfig);
	console.log("isInterrupted:", await session.isInterrupted());
	const result = await session.resume();
	if (!result) {
		console.log("nothing to resume?!");
		process.exit(1);
	}
	console.log(`\nRESUMED RESULT: ${result.status}`);
	console.log(`TEXT: ${result.text.slice(0, 300)}`);
	const stats = await session.stats();
	console.log(
		`lifetime: ${stats.steps} steps, $${stats.cost.toFixed(6)}, context ${stats.contextTokens} tokens`,
	);
	process.exit(result.status === "completed" ? 0 : 1);
} else {
	// orchestrate: start child, SIGKILL it mid-run, then resume in a new process
	const { rmSync } = await import("node:fs");
	rmSync(DIR, { recursive: true, force: true });
	console.log("── phase 1: start run, SIGKILL after 8s ──");
	// tsx's CLI re-spawns node as a child, so kill the whole process group —
	// killing just the wrapper leaves the real run alive and defeats the demo
	const child = spawn("./node_modules/.bin/tsx", ["playground/mvp/demo-restart.ts", "start"], {
		stdio: "inherit",
		detached: true,
	});
	await new Promise((r) => setTimeout(r, 8_000));
	if (!child.pid) throw new Error("child process failed to spawn");
	process.kill(-child.pid, "SIGKILL");
	console.log("── KILLED mid-run (simulated crash/deploy) ──");
	await new Promise((r) => setTimeout(r, 500));
	console.log("── phase 2: new process resumes from storage ──");
	const resume = spawn("./node_modules/.bin/tsx", ["playground/mvp/demo-restart.ts", "resume"], {
		stdio: "inherit",
	});
	const code = await new Promise<number>((r) => resume.on("exit", (c) => r(c ?? 1)));
	process.exit(code);
}
