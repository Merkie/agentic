import { createAgentic, createTools } from "../src/index.js";
import { tool } from "ai";
import { z } from "zod";
import { readJsonl } from "../src/jsonl.js";

const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) {
  console.error("Missing OPENROUTER_API_KEY in env");
  process.exit(1);
}

const start = Date.now();
const ts = () => `+${((Date.now() - start) / 1000).toFixed(2)}s`;
const log = (label: string, msg: string) => console.log(`[${ts()}] ${label} ${msg}`);

const tools = createTools(() => ({
  write_file: tool({
    description:
      "Write text content to a file path. This tool simulates a slow disk and takes ~5 seconds.",
    inputSchema: z.object({
      path: z.string().describe('Relative file path, e.g. "lorem.txt".'),
      content: z.string().describe("Full text contents to write."),
    }),
    execute: async ({ path, content }) => {
      log("tool", `write_file START path=${path} bytes=${content.length}`);
      await new Promise((resolve) => setTimeout(resolve, 5000));
      log("tool", `write_file DONE  path=${path}`);
      return { ok: true, path, bytesWritten: content.length };
    },
  }),
}));

const agentic = createAgentic({
  openRouterApiKey: apiKey,
  defaultModel: process.env.MODEL ?? "google/gemini-3.1-flash-lite-preview",
});

function attachEventLogger(session: ReturnType<typeof agentic.getSession>) {
  let turnCount = 0;
  session.onEvent((event) => {
    if (event.type === "text-start") {
      turnCount++;
      process.stdout.write(`\n[${ts()}] turn${turnCount} assistant: `);
    }
    if (event.type === "text-delta") process.stdout.write(event.text);
    if (event.type === "text-end") process.stdout.write("\n");
    if (event.type === "tool-input-start") {
      log("event", `tool-input-start ${event.toolName}`);
    }
    if (event.type === "tool-result") {
      log("event", `tool-result ${event.toolName}`);
    }
  });
}

function summarizeRunStarts(file: string) {
  const runStarts = readJsonl(file).filter((r) => r.kind === "run_start");
  console.log(`\n--- ${file} → ${runStarts.length} run_start record(s) ---`);
  runStarts.forEach((r: any, i) => {
    console.log(`  run #${i + 1}: ${r.inputMessages.length} input message(s)`);
    r.inputMessages.forEach((m: any, j: number) => {
      const c =
        typeof m.content === "string" ? m.content : JSON.stringify(m.content).slice(0, 80);
      console.log(`    [${j}] ${m.role}: ${c.slice(0, 90)}`);
    });
  });
}

// ============================================================================
// Phase A — 1 queued message
// ============================================================================
console.log("\n========================= PHASE A: 1 queued =========================");
const logA = `queue-test-A-${Date.now()}.jsonl`;
const sessionA = agentic.getSession({
  id: "queue-test-A",
  file: logA,
  system:
    "You are a concise assistant. When asked to write a file, call the write_file tool exactly once per requested file, then briefly confirm in one short sentence.",
  tools,
});
attachEventLogger(sessionA);

log("main", "send #1 (lorem.txt, 5s tool)");
const a1 = sessionA.send(
  "Write exactly 2 paragraphs of lorem ipsum to lorem.txt using the write_file tool.",
);
await new Promise((r) => setTimeout(r, 500));
log("main", "send #2 while turn 1 runs → should queue, run alone");
const a2 = sessionA.send(
  "Now write the single word 'queued!' to queued.txt using the write_file tool.",
);

const [a1r, a2r] = await Promise.all([a1, a2]);
log("main", `phase A done: a1=${a1r.finishReason} a2=${a2r.finishReason}`);
summarizeRunStarts(logA);

// ============================================================================
// Phase B — 3 queued messages, all batched into ONE follow-up turn
// ============================================================================
const phaseBStart = Date.now();
console.log("\n========================= PHASE B: 3 queued =========================");
const logB = `queue-test-B-${Date.now()}.jsonl`;
const sessionB = agentic.getSession({
  id: "queue-test-B",
  file: logB,
  system:
    "You are a concise assistant. When asked to write files, call the write_file tool once per file requested across ALL user messages in this turn, then briefly confirm in one short sentence.",
  tools,
});
attachEventLogger(sessionB);

log("main", "send #1 (alpha.txt, 5s tool)");
const b1 = sessionB.send(
  "Write the word 'alpha' to alpha.txt using the write_file tool.",
);
await new Promise((r) => setTimeout(r, 500));

log("main", "queue #2 (bravo.txt)");
const b2 = sessionB.send(
  "Also write the word 'bravo' to bravo.txt using the write_file tool.",
);
log("main", "queue #3 (charlie.txt)");
const b3 = sessionB.send(
  "Also write the word 'charlie' to charlie.txt using the write_file tool.",
);
log("main", "queue #4 (delta.txt)");
const b4 = sessionB.send(
  "Also write the word 'delta' to delta.txt using the write_file tool.",
);

const results = await Promise.all([b1, b2, b3, b4]);
const phaseBSec = ((Date.now() - phaseBStart) / 1000).toFixed(2);
log("main", `phase B done in ${phaseBSec}s: ${results.map((r) => r.finishReason).join(", ")}`);
console.log("  resolved runIds:");
console.log("    b1:", results[0].runId);
console.log("    b2:", results[1].runId);
console.log("    b3:", results[2].runId);
console.log("    b4:", results[3].runId);
console.log(
  `  b2/b3/b4 share runId? ${
    results[1].runId === results[2].runId && results[2].runId === results[3].runId
  }`,
);
console.log(`  b1 distinct from b2? ${results[0].runId !== results[1].runId}`);

summarizeRunStarts(logB);
