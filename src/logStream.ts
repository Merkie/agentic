import type { LanguageModelUsage, TextStreamPart, ToolSet } from "ai";
import chalk from "chalk";

type OpenRouterUsageCost = {
	cost?: number;
	costDetails?: {
		upstreamInferenceCost?: number | null;
	};
};

type OpenRouterModelResponse = {
	data?: {
		context_length?: unknown;
		top_provider?: {
			context_length?: unknown;
		};
	};
};

const openRouterContextLengthCache = new Map<string, Promise<number | null>>();

function getOpenRouterCost(part: TextStreamPart<ToolSet>) {
	if (part.type !== "finish-step") return null;

	const metadataUsage = part.providerMetadata?.openrouter?.usage as
		| OpenRouterUsageCost
		| undefined;
	const rawUsage = part.usage.raw as
		| {
				cost?: number;
				cost_details?: { upstream_inference_cost?: number | null };
		  }
		| undefined;

	const cost = metadataUsage?.cost ?? rawUsage?.cost;
	const upstreamInferenceCost =
		metadataUsage?.costDetails?.upstreamInferenceCost ??
		rawUsage?.cost_details?.upstream_inference_cost;

	// OpenRouter credit usage reports `cost`; BYOK usage can report `cost: 0`
	// with the real provider charge in `upstreamInferenceCost`.
	if (cost === 0 && upstreamInferenceCost != null) {
		return upstreamInferenceCost;
	}

	return cost ?? upstreamInferenceCost ?? null;
}

function formatCost(cost: number) {
	return `$${cost.toFixed(6)}`;
}

function formatTokens(tokens: number) {
	return tokens.toLocaleString("en-US");
}

function parsePositiveInteger(value: unknown) {
	if (typeof value === "number" && Number.isInteger(value) && value > 0) {
		return value;
	}

	if (typeof value === "string") {
		const parsed = Number(value);
		if (Number.isInteger(parsed) && parsed > 0) {
			return parsed;
		}
	}

	return null;
}

function getOpenRouterContextLengthFromResponse(payload: unknown) {
	const data = (payload as OpenRouterModelResponse).data;
	return (
		parsePositiveInteger(data?.context_length) ??
		parsePositiveInteger(data?.top_provider?.context_length)
	);
}

async function fetchOpenRouterContextLength(modelId: string) {
	const pathModelId = modelId.split("/").map(encodeURIComponent).join("/");
	const response = await fetch(
		`https://openrouter.ai/api/v1/model/${pathModelId}`,
	);

	if (!response.ok) {
		return null;
	}

	return getOpenRouterContextLengthFromResponse(await response.json());
}

function getOpenRouterContextLength(modelId: string) {
	const cached = openRouterContextLengthCache.get(modelId);
	if (cached) return cached;

	const promise = fetchOpenRouterContextLength(modelId).catch(() => null);
	openRouterContextLengthCache.set(modelId, promise);
	return promise;
}

function getModelId(part: TextStreamPart<ToolSet>) {
	if (part.type === "start-step") {
		const body = part.request.body;
		if (body && typeof body === "object" && "model" in body) {
			const model = body.model;
			if (typeof model === "string" && model.length > 0) {
				return model;
			}
		}
	}

	if (part.type === "finish-step") {
		return part.response.modelId;
	}

	return null;
}

function getUsageTokens(usage: LanguageModelUsage | undefined) {
	if (!usage) return null;

	// Context usage is the stopping-point conversation size estimate:
	// the last step's input tokens are the full history before the final response,
	// and its output tokens are what will be appended to that history. An exact
	// "next empty user message" count would require another provider tokenization
	// pass because the stream does not emit usage for hypothetical future calls.
	if (usage.inputTokens != null || usage.outputTokens != null) {
		return (usage.inputTokens ?? 0) + (usage.outputTokens ?? 0);
	}

	return usage.totalTokens ?? null;
}

function formatContextUsage(
	usage: LanguageModelUsage | undefined,
	contextWindowTokens: number | null,
) {
	const tokens = getUsageTokens(usage);
	if (tokens == null) return "context used ?";

	if (contextWindowTokens == null) {
		return `context used ${formatTokens(tokens)} tokens`;
	}

	const percentage = (tokens / contextWindowTokens) * 100;
	const formattedPercentage =
		percentage > 0 && percentage < 0.01 ? "<0.01" : percentage.toFixed(2);

	return `${formattedPercentage}% context used (${formatTokens(tokens)} / ${formatTokens(contextWindowTokens)} tokens)`;
}

/**
 * Pretty-prints every event coming off an AI SDK full stream so you can watch
 * the model think, talk, and call tools in real time — and see live token /
 * cost accounting when the model is served through OpenRouter (see
 * {@link createOpenRouter}).
 *
 * Pass it the `fullStream` from `streamText`:
 *
 * @example
 * ```ts
 * import { streamText } from "ai";
 * import { createOpenRouter, logStream } from "agentlib";
 *
 * const openrouter = createOpenRouter();
 *
 * const result = streamText({
 *   model: openrouter("openai/gpt-4o-mini"),
 *   prompt: "Explain quantum tunneling in one paragraph.",
 * });
 *
 * await logStream(result.fullStream);
 * ```
 */
export async function logStream(
	stream: AsyncIterable<TextStreamPart<ToolSet>>,
): Promise<void> {
	// Track which "channel" (reasoning vs. text) we're currently writing to so we
	// only print a header when it changes, and stream deltas onto the same line.
	let active: "reasoning" | "text" | null = null;
	let openRouterCost = 0;
	let hasOpenRouterCost = false;
	let latestStepUsage: LanguageModelUsage | undefined;
	let contextWindowPromise: Promise<number | null> | undefined;

	const endActive = () => {
		if (active) process.stdout.write("\n");
		active = null;
	};

	const startContextWindowFetch = (modelId: string | null) => {
		if (!modelId || contextWindowPromise) return;
		contextWindowPromise = getOpenRouterContextLength(modelId);
	};

	for await (const part of stream) {
		switch (part.type) {
			case "start-step":
				endActive();
				startContextWindowFetch(getModelId(part));
				console.log(chalk.dim("──────── step ────────"));
				break;

			case "finish-step": {
				latestStepUsage = part.usage;
				startContextWindowFetch(getModelId(part));
				const cost = getOpenRouterCost(part);
				if (cost != null) {
					openRouterCost += cost;
					hasOpenRouterCost = true;
				}
				break;
			}

			case "reasoning-delta":
				if (active !== "reasoning") {
					endActive();
					process.stdout.write(chalk.magenta.bold("🧠 reasoning  "));
					active = "reasoning";
				}
				process.stdout.write(chalk.magenta(part.text));
				break;

			case "text-delta":
				if (active !== "text") {
					endActive();
					process.stdout.write(chalk.white.bold("💬 message    "));
					active = "text";
				}
				process.stdout.write(chalk.white(part.text));
				break;

			case "tool-call":
				endActive();
				console.log(
					chalk.cyan.bold(`🔧 tool call  ${part.toolName}`) +
						chalk.cyan(`(${JSON.stringify(part.input)})`),
				);
				break;

			case "tool-result":
				endActive();
				console.log(
					chalk.green.bold(`✅ tool result ${part.toolName} `) +
						chalk.green(JSON.stringify(part.output)),
				);
				break;

			case "tool-error":
				endActive();
				console.log(
					chalk.red.bold(`❌ tool error ${part.toolName} `) +
						chalk.red(String(part.error)),
				);
				break;

			case "error":
				endActive();
				console.log(chalk.red.bold("‼️  stream error "), part.error);
				break;

			case "finish": {
				endActive();
				const contextWindowTokens = contextWindowPromise
					? await contextWindowPromise
					: null;
				console.log(
					chalk.dim(
						`\n── done (${part.finishReason}) · ${formatContextUsage(latestStepUsage, contextWindowTokens)}${hasOpenRouterCost ? ` · cost ${formatCost(openRouterCost)}` : ""} ──`,
					),
				);
				break;
			}
		}
	}
}

export default logStream;
