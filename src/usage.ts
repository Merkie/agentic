import type { LanguageModelUsage } from "ai";
import type { StepUsage, UsageTotals } from "./types.js";

// Per-step usage + cost extraction. Cost only exists when the model is served
// through OpenRouter (usage accounting), and it arrives in two shapes:
// camelCase on providerMetadata.openrouter.usage and snake_case on usage.raw.

interface OpenRouterUsageMeta {
	cost?: number;
	costDetails?: { upstreamInferenceCost?: number | null };
}

interface RawUsage {
	cost?: number;
	cost_details?: { upstream_inference_cost?: number | null };
}

/**
 * Reconcile OpenRouter's two cost fields into "what this step actually cost".
 * Paying with OpenRouter credits puts the charge in `cost`; BYOK requests
 * report the OpenRouter fee (often 0) in `cost` with the real provider charge
 * in `upstream_inference_cost` — so a BYOK step's true cost is the sum.
 */
export function reconcileBilledCost(
	cost: number | null,
	upstreamCost: number | null,
): number | null {
	if (cost === null && upstreamCost === null) return null;
	return (cost ?? 0) + (upstreamCost ?? 0);
}

/**
 * Extract normalized usage + cost from a finished step (the `onStepFinish`
 * payload or a `finish-step` stream part — both carry `usage` and
 * `providerMetadata`).
 */
export function extractStepUsage(step: {
	usage: LanguageModelUsage;
	providerMetadata?: Record<string, unknown>;
}): StepUsage {
	const usage = step.usage;
	const meta = (
		step.providerMetadata as { openrouter?: { usage?: OpenRouterUsageMeta } } | undefined
	)?.openrouter?.usage;
	const raw = (usage as { raw?: RawUsage }).raw;

	const cost = meta?.cost ?? raw?.cost ?? null;
	const upstreamCost =
		meta?.costDetails?.upstreamInferenceCost ?? raw?.cost_details?.upstream_inference_cost ?? null;

	const details = usage as {
		inputTokenDetails?: { cacheReadTokens?: number | null };
		outputTokenDetails?: { reasoningTokens?: number | null };
	};

	return {
		inputTokens: usage.inputTokens ?? null,
		outputTokens: usage.outputTokens ?? null,
		totalTokens: usage.totalTokens ?? null,
		cachedInputTokens: details.inputTokenDetails?.cacheReadTokens ?? null,
		reasoningTokens: details.outputTokenDetails?.reasoningTokens ?? null,
		cost,
		upstreamCost,
		billedCost: reconcileBilledCost(cost, upstreamCost),
	};
}

export function emptyTotals(): UsageTotals {
	return {
		inputTokens: 0,
		outputTokens: 0,
		totalTokens: 0,
		cachedInputTokens: 0,
		reasoningTokens: 0,
		cost: 0,
		steps: 0,
		contextTokens: null,
		contextWindow: null,
	};
}

/**
 * Fold one step into running totals. Cost and token counts accumulate over
 * time; `contextTokens` is NOT cumulative — it is the latest step's
 * input+output, i.e. the current size of the live conversation.
 */
export function addStepToTotals(totals: UsageTotals, step: StepUsage): UsageTotals {
	return {
		inputTokens: totals.inputTokens + (step.inputTokens ?? 0),
		outputTokens: totals.outputTokens + (step.outputTokens ?? 0),
		totalTokens: totals.totalTokens + (step.totalTokens ?? 0),
		cachedInputTokens: totals.cachedInputTokens + (step.cachedInputTokens ?? 0),
		reasoningTokens: totals.reasoningTokens + (step.reasoningTokens ?? 0),
		cost: totals.cost + (step.billedCost ?? 0),
		steps: totals.steps + 1,
		contextTokens: contextTokensOf(step) ?? totals.contextTokens,
		contextWindow: totals.contextWindow,
	};
}

/** Live conversation size after a step: its input tokens + what it appended. */
export function contextTokensOf(step: StepUsage): number | null {
	if (step.inputTokens === null && step.outputTokens === null) {
		return step.totalTokens;
	}
	return (step.inputTokens ?? 0) + (step.outputTokens ?? 0);
}
