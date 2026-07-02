import type { ClassifiedFailure, FailureKind, SerializedError } from "./types.js";

// Failure classification for OpenRouter/AI SDK errors. The taxonomy is ported
// from a production system (halo-cmo's agentResilience) where every needle was
// added because a real run failed on it, plus OpenRouter's documented
// `error.metadata.error_type` codes and opencode's retry rules. Three buckets:
//
//   transient         → retry with backoff (server-directed wait wins)
//   context-overflow  → a compaction signal; retrying replays the same
//                       oversized state forever
//   fatal             → deterministic (auth, billing, policy, malformed
//                       request); retrying burns credits for nothing

// ── context-window overflow ─────────────────────────────────────────────

const CONTEXT_OVERFLOW_PATTERNS = [
	/maximum context length/i,
	/context length is/i,
	/context[_ ]length[_ ]exceeded/i,
	/exceeds the context window/i,
	/exceeds (?:the )?(?:model'?s )?maximum context length/i,
	/exceeds (?:the )?maximum allowed input length/i,
	/input token count.*exceeds the maximum/i,
	/input length.*exceeds.*context length/i,
	/prompt is too long/i,
	/prompt too long/i,
	/request_too_large/i,
	/request entity too large/i,
	/reduce the length of (?:the )?(?:messages|input|prompt)/i,
	/too many tokens/i,
	/token limit exceeded/i,
	/model_context_window_exceeded/i,
];

// Rate-limit phrasings that contain overflow-ish words ("too many …") — must
// win over the overflow patterns so a 429 is retried, not compacted.
const NON_OVERFLOW_PATTERNS = [/rate.?limit/i, /too many requests/i, /throttl/i];

export function isContextOverflow(haystack: string): boolean {
	if (NON_OVERFLOW_PATTERNS.some((p) => p.test(haystack))) return false;
	return CONTEXT_OVERFLOW_PATTERNS.some((p) => p.test(haystack));
}

// ── deterministic (fatal) signals ───────────────────────────────────────

const FATAL_NEEDLES = [
	// billing / credits — replaying against a dead balance is pure spin
	"requires more credits",
	"fewer max_tokens",
	"payment required",
	"insufficient_quota",
	"insufficient credits",
	"insufficient funds",
	"insufficient balance",
	"available balance",
	"out of credits",
	"out of budget",
	"quota exceeded",
	"billing",
	// auth
	"invalid api key",
	"invalid_api_key",
	"authentication",
	"unauthorized",
	"no auth credentials",
	// policy / moderation
	"content_policy",
	"content policy",
	"safety policy",
	"content_policy_violation",
	"moderation",
	"refusal",
	// malformed request — replays identically. OpenRouter wraps provider 400s
	// as generic "provider returned error" (a transient needle), so these must
	// match first or a hopeless request retries forever.
	"invalid_prompt",
	"invalid_request",
	"param incorrect",
	"missing a function name",
];

// OpenRouter's stable typed error codes (error.metadata.error_type) — more
// reliable than HTTP status when present.
const OPENROUTER_FATAL_TYPES = new Set([
	"invalid_request",
	"content_policy_violation",
	"refusal",
	"context_length_exceeded", // handled as overflow below, listed for clarity
	"max_tokens_exceeded",
]);
const OPENROUTER_TRANSIENT_TYPES = new Set([
	"rate_limit_exceeded",
	"provider_overloaded",
	"provider_unavailable",
	"timeout",
	"server",
	"unmapped",
]);

// ── transient signals ───────────────────────────────────────────────────

const TRANSIENT_NEEDLES = [
	// transport / socket
	"und_err_socket",
	"econnreset",
	"etimedout",
	"econnrefused",
	"eai_again",
	"socket hang up",
	"other side closed",
	"terminated",
	"fetch failed",
	"network error",
	"connection error",
	"connection refused",
	"connection reset",
	"upstream connect",
	"reset before headers",
	"stream ended before",
	"response stream error",
	"sse read timed out",
	"sse stream produced no bytes",
	"header timeout",
	"headers timed out",
	"timed out",
	"timeout",
	// provider-side transient (OpenRouter / upstream gateways)
	"provider_unavailable",
	"provider_overloaded",
	"provider returned error",
	"provider returned an error",
	"server_error",
	"server is overloaded",
	"upstream idle timeout",
	"idle timeout exceeded",
	"bad gateway",
	"gateway timeout",
	"service unavailable",
	"temporarily unavailable",
	"overloaded",
	"server error",
	"internal error",
	"internal server error",
	"rate limit",
	"rate_limit",
	"too many requests",
	"too_many_requests",
	"exhausted",
];

// Numeric status matched as a field, not a bare substring, so token counts and
// prices can't trip it.
const RETRYABLE_HTTP_STATUS =
	/"(?:status|statuscode|status_code|code)":\s*"?(?:408|429|529|5\d\d)"?(?=[,}\]])/;
const RETRYABLE_STATUS_TEXT = /\b(?:http|status|code|error)\D{0,12}(?:408|429|529|5\d\d)\b/;
const RETRYABLE_FLAG = /"(?:isretryable|retryable)":\s*true/;

// AI SDK `AI_NoOutputGeneratedError` — thrown at stream flush when the
// provider stream errored before producing anything. Deterministic no-output
// (overflow, credits) is excluded above, so reaching this means a retryable
// provider drop.
const NO_OUTPUT_GENERATED = /ai_nooutputgeneratederror|no output generated/;

// ── error serialization ─────────────────────────────────────────────────

export function serializeError(err: unknown): SerializedError {
	if (err instanceof Error) {
		const anyErr = err as Error & {
			statusCode?: number;
			status?: number;
			responseBody?: unknown;
			data?: unknown;
			cause?: unknown;
		};
		return {
			name: err.name,
			message: err.message,
			status: anyErr.statusCode ?? anyErr.status,
			detail: safeDetail(anyErr.responseBody ?? anyErr.data ?? anyErr.cause),
		};
	}
	if (err && typeof err === "object") {
		const anyErr = err as { message?: unknown; code?: unknown; name?: unknown };
		return {
			name: typeof anyErr.name === "string" ? anyErr.name : undefined,
			message:
				typeof anyErr.message === "string" && anyErr.message.trim()
					? anyErr.message
					: safeStringify(err),
			status: typeof anyErr.code === "number" || typeof anyErr.code === "string" ? anyErr.code : undefined,
			detail: safeDetail(err),
		};
	}
	return { message: err === undefined || err === null ? "Unknown error" : String(err) };
}

function safeStringify(value: unknown): string {
	try {
		return JSON.stringify(value) ?? String(value);
	} catch {
		return String(value);
	}
}

function safeDetail(value: unknown): unknown {
	if (value === undefined || value === null) return undefined;
	const s = safeStringify(value);
	return s.length > 4000 ? s.slice(0, 4000) : value;
}

// ── Retry-After extraction ──────────────────────────────────────────────

function retryAfterMsFrom(err: unknown): number | undefined {
	const headers = (err as { responseHeaders?: Record<string, string> })?.responseHeaders;
	if (!headers) return undefined;
	const lower: Record<string, string> = {};
	for (const [k, v] of Object.entries(headers)) lower[k.toLowerCase()] = v;
	const ms = Number.parseFloat(lower["retry-after-ms"] ?? "");
	if (Number.isFinite(ms) && ms > 0) return ms;
	const raw = lower["retry-after"];
	if (!raw) return undefined;
	const seconds = Number.parseFloat(raw);
	if (Number.isFinite(seconds) && seconds > 0) return seconds * 1000;
	const date = Date.parse(raw);
	if (!Number.isNaN(date)) {
		const delta = date - Date.now();
		if (delta > 0) return delta;
	}
	return undefined;
}

// ── the classifier ──────────────────────────────────────────────────────

/**
 * Decide what to do with a failed model call or errored stream. Inspects the
 * whole error object (message, status, OpenRouter `error_type`, nested cause)
 * so it works on thrown APICallErrors, stream `error` chunks, and serialized
 * failure records alike.
 */
export function classifyFailure(err: unknown): ClassifiedFailure {
	const error = serializeError(err);
	const haystack = safeStringify({
		error,
		raw: err instanceof Error ? { name: err.name, message: err.message } : err,
	}).toLowerCase();

	const kind = classifyHaystack(haystack, err);
	return { kind, error, retryAfterMs: kind === "transient" ? retryAfterMsFrom(err) : undefined };
}

function classifyHaystack(haystack: string, err: unknown): FailureKind {
	// OpenRouter typed code wins when present.
	const typeMatch = haystack.match(/"error_type":\s*"([a-z_]+)"/);
	if (typeMatch) {
		const t = typeMatch[1];
		if (t === "context_length_exceeded") return "context-overflow";
		if (OPENROUTER_TRANSIENT_TYPES.has(t)) return "transient";
		if (OPENROUTER_FATAL_TYPES.has(t)) return "fatal";
	}

	if (isContextOverflow(haystack)) return "context-overflow";
	if (FATAL_NEEDLES.some((n) => haystack.includes(n))) return "fatal";
	if (/"(?:status|statuscode|status_code|code)":\s*"?40[0134]"?(?=[,}\]])/.test(haystack)) {
		return "fatal";
	}

	const isRetryable = (err as { isRetryable?: boolean })?.isRetryable;
	if (isRetryable === true) return "transient";

	if (
		RETRYABLE_HTTP_STATUS.test(haystack) ||
		RETRYABLE_STATUS_TEXT.test(haystack) ||
		RETRYABLE_FLAG.test(haystack)
	) {
		return "transient";
	}
	if (NO_OUTPUT_GENERATED.test(haystack)) return "transient";
	if (TRANSIENT_NEEDLES.some((n) => haystack.includes(n))) return "transient";

	return "fatal";
}

/** Best-effort human-readable message from a stream `error` part. */
export function describeError(e: unknown): string {
	if (e instanceof Error) return e.message;
	if (e && typeof e === "object") {
		const m = (e as { message?: unknown }).message;
		if (typeof m === "string" && m.trim()) return m;
		return safeStringify(e);
	}
	return e === undefined || e === null ? "Model stream ended with an error" : String(e);
}
