import type { AgenticUsageLike } from "./types.js";

export function normalizeUsage(usage: AgenticUsageLike | undefined): {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
} {
  const inputTokens = numberOrZero(usage?.inputTokens ?? usage?.promptTokens);
  const outputTokens = numberOrZero(usage?.outputTokens ?? usage?.completionTokens);
  const totalTokens = numberOrZero(usage?.totalTokens) || inputTokens + outputTokens;
  return { inputTokens, outputTokens, totalTokens };
}

export function calculateContextUsage(opts: {
  usedTokens: number;
  contextLength: number | null;
}): {
  contextLength: number | null;
  usedTokens: number;
  usedPct: number | null;
  remainingPct: number | null;
} {
  if (!opts.contextLength || opts.contextLength <= 0) {
    return {
      contextLength: opts.contextLength,
      usedTokens: opts.usedTokens,
      usedPct: null,
      remainingPct: null,
    };
  }
  const usedPct = Math.min(opts.usedTokens / opts.contextLength, 1);
  return {
    contextLength: opts.contextLength,
    usedTokens: opts.usedTokens,
    usedPct,
    remainingPct: Math.max(1 - usedPct, 0),
  };
}

export function extractOpenRouterStepCost(step: any): number {
  const cost = step?.providerMetadata?.openrouter?.usage?.cost;
  return typeof cost === "number" && Number.isFinite(cost) && cost > 0 ? cost : 0;
}

function numberOrZero(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}
