import type { RetryConfig } from "./types.js";

export interface ResolvedRetryConfig {
	maxAttempts: number;
	baseDelayMs: number;
	maxDelayMs: number;
	maxElapsedMs: number;
}

// Defaults tuned for "never fail a run that a short wait would have saved,
// never spin for hours": 6 attempts of capped exponential backoff is ~3.5
// minutes of waiting, and the elapsed budget hard-stops a run whose retries
// (plus server-directed Retry-After waits) drag past 15 minutes.
export function resolveRetryConfig(config?: RetryConfig): ResolvedRetryConfig {
	return {
		maxAttempts: config?.maxAttempts ?? 6,
		baseDelayMs: config?.baseDelayMs ?? 2_000,
		maxDelayMs: config?.maxDelayMs ?? 60_000,
		maxElapsedMs: config?.maxElapsedMs ?? 15 * 60_000,
	};
}

/**
 * Delay before retry `attempt` (1-based). A server-directed Retry-After wins
 * over exponential backoff; both are capped by `maxDelayMs`.
 */
export function retryDelayMs(
	attempt: number,
	config: ResolvedRetryConfig,
	retryAfterMs?: number,
): number {
	if (retryAfterMs !== undefined && retryAfterMs > 0) {
		return Math.min(retryAfterMs, config.maxDelayMs);
	}
	const exp = config.baseDelayMs * 2 ** Math.max(0, attempt - 1);
	return Math.min(exp, config.maxDelayMs);
}

/** Abort-aware sleep. Rejects with the abort reason if the signal fires. */
export function wait(ms: number, signal?: AbortSignal): Promise<void> {
	if (ms <= 0) return Promise.resolve();
	return new Promise<void>((resolve, reject) => {
		if (signal?.aborted) {
			reject(signal.reason ?? new Error("Aborted"));
			return;
		}
		let settled = false;
		const onAbort = () => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			reject(signal?.reason ?? new Error("Aborted"));
		};
		// Deliberately NOT unref'd: this wait is often the only thing keeping a
		// retrying run alive — unref'ing it lets a standalone process exit
		// mid-backoff and silently abandon the run.
		const timer = setTimeout(() => {
			if (settled) return;
			settled = true;
			signal?.removeEventListener("abort", onAbort);
			resolve();
		}, ms);
		signal?.addEventListener("abort", onAbort, { once: true });
	});
}
