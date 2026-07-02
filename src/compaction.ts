import { generateText, type LanguageModel, type ModelMessage } from "ai";
import { sanitizeConversation } from "./sanitize.js";
import type { CompactionConfig, StepUsage } from "./types.js";
import { extractStepUsage } from "./usage.js";

export interface ResolvedCompactionConfig {
	limit: number;
	keepRecent: number;
	prompt: string;
	model?: string;
}

export const DEFAULT_COMPACTION_PROMPT = `You are writing a hand-off summary of the conversation so far. The chat history will be cleared and replaced with your summary, so anything you leave out is gone forever. Write for the assistant who picks the conversation back up.

Cover, in order:
1. CONTEXT — who the user is, what they want overall, standing instructions and preferences they've expressed.
2. WHAT HAPPENED — what has been discussed, decided, and produced so far. Preserve exact names, ids, file paths, URLs, numbers, and tool results that may matter later.
3. ACTIVE TASK — what was being worked on right now, its exact state, and what remains.
4. OPEN THREADS — anything promised, pending, or unresolved.

If the history already contains an earlier summary, carry its still-relevant facts forward. Output only the summary text, no preamble.`;

export function resolveCompactionConfig(config?: CompactionConfig): ResolvedCompactionConfig {
	return {
		limit: config?.limit ?? 0.5,
		keepRecent: config?.keepRecent ?? 8,
		prompt: config?.prompt ?? DEFAULT_COMPACTION_PROMPT,
		model: config?.model,
	};
}

/**
 * True when the live conversation has crossed the compaction threshold.
 * `contextTokens` is the latest step's provider-reported input+output — real
 * tokenizer counts, not message counts or char/4 estimates.
 */
export function shouldCompact(
	contextTokens: number | null,
	contextWindow: number | null,
	limit: number,
): boolean {
	if (limit <= 0 || contextTokens === null || contextWindow === null) return false;
	return contextTokens >= Math.floor(contextWindow * limit);
}

export interface CompactionResult {
	/** The new replay base: [summary-as-user-message, ...recent tail]. */
	messages: ModelMessage[];
	summary: string;
	usage: StepUsage;
}

/**
 * Summarize everything but the recent tail into a hand-off message. The
 * result becomes the session's new replay base; the full pre-compaction
 * ledger is retained by storage for audit/debugging.
 */
export async function runCompaction(options: {
	messages: ModelMessage[];
	model: LanguageModel;
	config: ResolvedCompactionConfig;
	abortSignal?: AbortSignal;
}): Promise<CompactionResult> {
	const { messages, config } = options;
	const keep = Math.min(config.keepRecent, Math.max(0, messages.length - 1));
	// Sanitize each slice so the cut can't strand half of a tool-call/result
	// pair on either side of the boundary.
	const head = sanitizeConversation(messages.slice(0, messages.length - keep)).messages;
	const tail = sanitizeConversation(keep > 0 ? messages.slice(messages.length - keep) : []).messages;

	const result = await generateText({
		model: options.model,
		system: config.prompt,
		messages: [
			...head,
			{
				role: "user",
				content:
					"Write the hand-off summary of the conversation above now, following your instructions.",
			},
		],
		abortSignal: options.abortSignal,
	});

	const summary = result.text.trim();
	const summaryMessage: ModelMessage = {
		role: "user",
		content: `<conversation-summary>\nThe earlier chat history was compacted into this summary. Treat it as ground truth and continue the conversation (and any active task) from here.\n\n${summary}\n</conversation-summary>`,
	};

	return {
		messages: [summaryMessage, ...tail],
		summary,
		usage: extractStepUsage({
			usage: result.usage,
			providerMetadata: result.providerMetadata as Record<string, unknown> | undefined,
		}),
	};
}
