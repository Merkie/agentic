import "dotenv/config";
import { existsSync } from "node:fs";
import { createAgentic, createTools } from "../src/index.js";
import { tool } from "ai";
import { z } from "zod";

const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) {
  console.error("Missing OPENROUTER_API_KEY in env");
  process.exit(1);
}

const prompt = process.argv.slice(2).join(" ").trim();
if (!prompt) {
  console.error('Usage: pnpm dev:send-message -- "your message"');
  process.exit(1);
}

const logFile = process.env.CHAT_LOG ?? "chat.jsonl";
const model = process.env.MODEL ?? "google/gemini-3.1-flash-lite-preview";

const tools = createTools(() => ({
  check_weather: tool({
    description: "Check deterministic demo weather for a city.",
    inputSchema: z.object({
      city: z.string().describe('City name, e.g. "Tokyo" or "San Francisco".'),
    }),
    execute: async ({ city }) => {
      return { city, condition: "clear skies", humidityPct: 42 };
    },
  }),
}));

const agentic = createAgentic({
  openRouterApiKey: apiKey,
  defaultModel: model,
});

const session = agentic.getSession({
  id: "cli",
  logFile,
  model,
  system:
    "You are a concise assistant. Use tools when they help, then answer directly.",
  tools,
});

console.log(existsSync(logFile) ? `(continuing ${logFile})` : `(new ${logFile})`);
console.log(`\n[user] ${prompt}\n`);

let assistantOpen = false;
session.onEvent((event) => {
  switch (event.type) {
    case "text-start":
      process.stdout.write("[assistant] ");
      assistantOpen = true;
      break;
    case "text-delta":
      process.stdout.write(event.text);
      break;
    case "text-end":
      if (assistantOpen) process.stdout.write("\n");
      assistantOpen = false;
      break;
    case "tool-input-start":
      if (assistantOpen) process.stdout.write("\n");
      assistantOpen = false;
      process.stdout.write(`[-> ${event.toolName}]\n`);
      break;
    case "tool-result":
      process.stdout.write(`[<- ${event.toolName} ${JSON.stringify(event.output)}]\n`);
      break;
  }
});

const result = await session.send(prompt);
console.log(`\n(saved ${result.logFile}, finish=${result.finishReason})`);
