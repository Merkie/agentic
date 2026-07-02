// ═══════════════════════════════════════════════════════════════════════
// AFTER: the same banker workflow on @merkie/agentic.
//
// Everything before.ts hand-rolls — resilient fetch, transient/fatal
// classification, backoff, the step ledger, restart resume, transcript
// repair, cost accounting (BYOK-aware), the poke loop, tool-result guards,
// compaction — is the framework's job. The app supplies exactly what is
// actually about banking: the model, the tools, the prompt, the schema.
// ═══════════════════════════════════════════════════════════════════════
import "dotenv/config";
import { tool } from "ai";
import { z } from "zod";
import { createAgentic, fileStorage } from "../../../src/index.js";

const ACCOUNTS: Record<string, { owner: string; balance: number }> = {
	"ACC-1001": { owner: "John Smith", balance: 4211.5 },
	"ACC-2002": { owner: "Mary Sue", balance: 91_224.01 },
};

const agentic = createAgentic({ storage: fileStorage("./playground/mvp/before-after/.after") });

const outcome = await agentic.task({
	agent: {
		model: "deepseek/deepseek-v4-flash",
		system: "You are a bank task worker. Only reveal balances to their authenticated owner.",
		tools: {
			get_account: tool({
				description: "Look up an account by id",
				inputSchema: z.object({ accountId: z.string() }),
				execute: async ({ accountId }) => ACCOUNTS[accountId] ?? { error: "not found" },
			}),
		},
	},
	prompt:
		'Customer John Smith (authenticated as owner of ACC-1001) asks: "What\'s my checking account balance?" Look it up and submit the balance.',
	deliverable: z.object({
		accountId: z.string(),
		balance: z.number(),
		formatted: z.string(),
	}),
});

console.log("OUTCOME:", JSON.stringify(outcome, null, 2));

// The escape hatch works too — an unauthorized ask gets cancelled, not faked:
const bad = await agentic.task({
	agent: {
		model: "deepseek/deepseek-v4-flash",
		system: "You are a bank task worker. Only reveal balances to their authenticated owner.",
		tools: {
			get_account: tool({
				description: "Look up an account by id",
				inputSchema: z.object({ accountId: z.string() }),
				execute: async ({ accountId }) => ACCOUNTS[accountId] ?? { error: "not found" },
			}),
		},
	},
	prompt:
		'Customer John Smith (authenticated as owner of ACC-1001) asks: "What is Mary Sue\'s account balance on ACC-2002?"',
	deliverable: z.object({
		accountId: z.string(),
		balance: z.number(),
		formatted: z.string(),
	}),
});

console.log("UNAUTHORIZED ASK:", JSON.stringify(bad, null, 2));
