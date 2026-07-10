// Multimodal validated task. Pass an image path and optional OpenRouter model.
import "dotenv/config";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import { createAgentic, memoryStorage } from "../../src/index.js";

const imagePath = process.argv[2];
if (!imagePath) {
	throw new Error("Usage: tsx demo-media-task.ts <image-path> [model]");
}
const image = new Uint8Array(await readFile(path.resolve(imagePath)));
const agentic = createAgentic({ storage: memoryStorage(), logs: true });
const outcome = await agentic.task({
	agent: { model: process.argv[3] ?? "google/gemini-3.1-flash-lite" },
	prompt: [
		{
			role: "user",
			content: [
				{ type: "text", text: "Classify this real-estate image and submit the result." },
				{ type: "image", image, mediaType: "image/jpeg" },
			],
		},
	],
	deliverable: z.object({
		scene: z.string(),
		indoor: z.boolean(),
		description: z.string().min(10),
	}),
});

console.log("\nOUTCOME:", JSON.stringify(outcome, null, 2));
if (outcome.status !== "submitted") process.exit(1);
