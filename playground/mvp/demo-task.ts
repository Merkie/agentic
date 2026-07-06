// Workflow task demo: guaranteed outcome, zod-validated deliverable through
// the submit tool (self-healing structured output), poke on early stream end.
import "dotenv/config";
import { z } from "zod";
import { createAgentic, memoryStorage } from "../../src/index.js";

const agentic = createAgentic({
	storage: memoryStorage(),
	logs: true,
});

const outcome = await agentic.task({
	agent: { model: process.argv[2] ?? "deepseek/deepseek-v4-flash" },
	prompt: `Analyze the sentiment of this review: "The product arrived broken but customer service replaced it within a day, overall I'm happy."

TEST INSTRUCTION: on your FIRST submit attempt, deliberately pass confidence as the string "high" (wrong type) so we can verify validation repair. Correct it after the tool rejects it.`,
	deliverable: z.object({
		sentiment: z.enum(["positive", "negative", "neutral"]),
		confidence: z.number().min(0).max(1),
		keywords: z.array(z.string()).min(2).max(5),
	}),
});

console.log("\nOUTCOME:", JSON.stringify(outcome, null, 2));
if (outcome.status !== "submitted") process.exit(1);
