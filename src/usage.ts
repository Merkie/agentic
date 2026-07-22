import type { LanguageModelUsage } from "ai";
import type { StepUsage, UsageTotals } from "./types.js";

// Per-step usage + cost extraction. Cost only exists when the model is served
// through OpenRouter (usage accounting), and it arrives in two shapes:
// camelCase on providerMetadata.openrouter.usage and snake_case on usage.raw.
// The `is_byok` discriminator ONLY survives on the raw snake_case shape —
// @openrouter/ai-sdk-provider (≤2.3.3) whitelists what it copies into the
// camelCase metadata and drops it — so extraction must read both.

interface OpenRouterUsageMeta {
	cost?: number;
	costDetails?: { upstreamInferenceCost?: number | null };
	isByok?: boolean;
}

interface RawUsage {
	cost?: number;
	cost_details?: { upstream_inference_cost?: number | null };
	is_byok?: boolean;
}

/**
 * Reconcile OpenRouter's cost fields into "what this step actually cost".
 * Which field carries the real charge depends on the billing regime, which
 * OpenRouter reports via `is_byok`:
 *
 * - BYOK (`isByok === true`): `cost` is OpenRouter's fee (often 0) and the
 *   real provider charge is in `upstream_inference_cost` — the true cost is
 *   the sum of the two.
 * - Credits (`isByok === false`): `cost` is the full charge. Since mid-2026
 *   OpenRouter also mirrors what it paid the provider into
 *   `upstream_inference_cost` on credits requests (informational, not an
 *   extra charge), so summing would double-count — the true cost is `cost`
 *   alone.
 * - Unknown (`isByok` missing — pre-`is_byok` payloads or models not served
 *   through OpenRouter): never sum. Take `cost`, falling back to
 *   `upstreamCost`; doubling a mirrored credits charge is worse than
 *   undercounting a fee-only BYOK figure.
 */
export function reconcileBilledCost(
	cost: number | null,
	upstreamCost: number | null,
	isByok: boolean | null = null,
): number | null {
	if (cost === null && upstreamCost === null) return null;
	if (isByok === true) return (cost ?? 0) + (upstreamCost ?? 0);
	return cost ?? upstreamCost;
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
	const isByok = meta?.isByok ?? raw?.is_byok ?? null;

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
		isByok,
		billedCost: reconcileBilledCost(cost, upstreamCost, isByok),
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
