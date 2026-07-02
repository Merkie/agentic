// Per-model metadata from OpenRouter, used for compaction math (context
// window) and cost estimation. Fetched lazily and memoized per model id.

interface ModelEndpointResponse {
	data?: {
		context_length?: unknown;
		top_provider?: { context_length?: unknown };
	};
}

const contextLengthCache = new Map<string, Promise<number | null>>();

function positiveInt(value: unknown): number | null {
	if (typeof value === "number" && Number.isInteger(value) && value > 0) return value;
	if (typeof value === "string") {
		const parsed = Number(value);
		if (Number.isInteger(parsed) && parsed > 0) return parsed;
	}
	return null;
}

async function fetchContextLength(modelId: string): Promise<number | null> {
	const path = modelId.split("/").map(encodeURIComponent).join("/");
	const response = await fetch(`https://openrouter.ai/api/v1/model/${path}`);
	if (!response.ok) return null;
	const payload = (await response.json()) as ModelEndpointResponse;
	return (
		positiveInt(payload.data?.context_length) ??
		positiveInt(payload.data?.top_provider?.context_length)
	);
}

/**
 * The model's context window in tokens, from OpenRouter's model endpoint.
 * Memoized for the process lifetime; resolves null when unknown (unknown
 * model, network failure) — callers must treat null as "cannot compact by
 * fraction".
 */
export function getContextWindow(modelId: string): Promise<number | null> {
	const cached = contextLengthCache.get(modelId);
	if (cached) return cached;
	const promise = fetchContextLength(modelId).catch(() => null);
	contextLengthCache.set(modelId, promise);
	return promise;
}

/** Test hook / offline override. */
export function setContextWindow(modelId: string, tokens: number | null): void {
	contextLengthCache.set(modelId, Promise.resolve(tokens));
}
