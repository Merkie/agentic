// Per-model metadata from OpenRouter, used for compaction math (context
// window) and cost estimation. Fetched lazily and memoized per model id.

interface ModelEndpointResponse {
	data?: {
		context_length?: unknown;
		top_provider?: { context_length?: unknown };
	};
}

const contextLengthCache = new Map<string, Promise<number | null>>();
const MODEL_METADATA_TIMEOUT_MS = 5_000;

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
	const response = await fetch(`https://openrouter.ai/api/v1/model/${path}`, {
		signal: AbortSignal.timeout(MODEL_METADATA_TIMEOUT_MS),
	});
	if (!response.ok) {
		// A missing model is a stable miss. Any other non-success may be
		// operational or temporary, so let getContextWindow evict and retry it.
		if (response.status === 404) return null;
		throw new Error(`OpenRouter model metadata request failed (${response.status})`);
	}
	const payload = (await response.json()) as ModelEndpointResponse;
	return (
		positiveInt(payload.data?.context_length) ??
		positiveInt(payload.data?.top_provider?.context_length)
	);
}

/**
 * The model's context window in tokens, from OpenRouter's model endpoint.
 * Successful lookups (including stable "unknown model" misses) are memoized
 * for the process lifetime. Transient HTTP/network/parse failures resolve null
 * for that call but are evicted so the next call can retry. Callers must treat
 * null as "cannot compact by fraction".
 */
export function getContextWindow(modelId: string): Promise<number | null> {
	const cached = contextLengthCache.get(modelId);
	if (cached) return cached;
	const request = fetchContextLength(modelId);
	const promise = request.catch(() => null);
	contextLengthCache.set(modelId, promise);
	void request.catch(() => {
		if (contextLengthCache.get(modelId) === promise) contextLengthCache.delete(modelId);
	});
	return promise;
}

/** Test hook / offline override. */
export function setContextWindow(modelId: string, tokens: number | null): void {
	contextLengthCache.set(modelId, Promise.resolve(tokens));
}
